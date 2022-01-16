import json
import os
import re
import glob
from datetime import datetime

import numpy as np
from numpy.lib.utils import source
import pandas as pd
from classification import FakeNewsNetDataset, FakeNewsNetDatasetTruncated, FakeNewsNetDatasetMetadata

PATH_TO_DATASET = '/run/media/philipp/729B4F335334713C/FakeNewsNet/code/fakenewsnet_dataset/'
PATH_TO_DATASET_CSV = '/run/media/philipp/729B4F335334713C/FakeNewsNet/dataset/'

POLITIFACT = "politifact"
GOSSIPCOP = "gossipcop"

NO_SUMMARY = "basic"
ABSTRACTIVE_SUMMARY = "abstractive"
EXTRACTIVE_SUMMARY = "extractive"

# Some articles got no text -> check whether a corresponding json exists or the text is empty


def check_if_file_exists(identifier, label, type_):
    if label == 0:
        path = os.path.join(
            PATH_TO_DATASET, type_, "real", identifier, "news content.json")
    else:
        path = os.path.join(
            PATH_TO_DATASET, type_, "fake", identifier, "news content.json")
    if(os.path.isfile(path)):
        with open(path) as f:
            data = json.load(f)
            data = json.load(f)
        if data['text'] != "":
            return True
        return False
    else:
        return False


def get_split(text, split_length, stride_length=50):
    l_total = []
    l_partial = []
    text_length = len(text.split())
    partial_length = split_length - stride_length
    if text_length//partial_length > 0:
        n = text_length//partial_length
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_partial = text.split()[:split_length]
            l_total.append(" ".join(l_partial))
        else:
            l_partial = text.split()[w*partial_length:w *
                                     partial_length + split_length]
            l_total.append(" ".join(l_partial))
    return l_total


def prep_dataframe(path_to_csv):
    df_real = pd.read_csv(path_to_csv+"_real.csv")
    df_real['label'] = 0
    df_fake = pd.read_csv(path_to_csv+"_fake.csv")
    df_fake['label'] = 1
    dfs = [df_real, df_fake]
    df = pd.concat(dfs)
    if(POLITIFACT in path_to_csv):
        df['type'] = POLITIFACT
    else:
        df['type'] = GOSSIPCOP
    return df


def get_sources(root_dir, news_dataframe):
    sources = []
    for idx in range(len(news_dataframe)):
        path = ""
        if news_dataframe.iloc[idx]['label'] == 0:
            path = os.path.join(root_dir, news_dataframe.iloc[idx]['type'],
                                'real', news_dataframe.iloc[idx]['id'], "news content.json")
        else:
            path = os.path.join(root_dir, news_dataframe.iloc[idx]['type'],
                                'fake', news_dataframe.iloc[idx]['id'], "news content.json")
        with open(path) as f:
            data = json.load(f)
        sources.append(data['source'])

    return sources

# Gets the encodings of each text - if it is longer than the maximium token length of BERT the mean embeddings instead


def get_news_content_encodings(root_dir, news_dataframe, tokenizer, summary, is_use_sources, is_use_authors, is_use_tweets, is_use_retweets):
    encodings = []
    labels = []
    sources = []
    authors = []
    retweet_counts = []
    tweet_embeddings = []
    for idx in range(len(news_dataframe)):
        path = ""
        if news_dataframe.iloc[idx]['label'] == 0:
            root_path = os.path.join(root_dir, news_dataframe.iloc[idx]['type'],
                                     'real', news_dataframe.iloc[idx, 0])
            path = os.path.join(root_path, "news content.json")
        else:
            root_path = os.path.join(root_dir, news_dataframe.iloc[idx]['type'],
                                     'fake', news_dataframe.iloc[idx, 0])
            path = os.path.join(root_path, "news content.json")
        if(summary == ABSTRACTIVE_SUMMARY):
            path = path[:-5] + "_summary_abstractive.json"
        if(summary == EXTRACTIVE_SUMMARY):
            path = path[:-5] + "_summary_extractive.json"
        with open(path) as f:
            data = json.load(f)

        sum_text = data['title'] + ". " + data['text']
                    
        if is_use_authors:
            if(data['authors']):
                authors.append(",".join(data["authors"]))
            else:
                authors.append("")

            authors.append(get_tweet_authors(root_path))

        if is_use_sources:
            if(data['source'] == "https://web.archive.org"):
                # findall & search behave different -> later ignores capturing groups and only returns the match - to circumvent this for findall use non capturing groups (?:)
                source = re.findall(
                    "https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}", data['url'])
                sources.append(source[1])
            else:
                sources.append(data['source'])

        if is_use_retweets:
            retweet_counts.append(get_retweet_count(root_path))

        # Split longer texts into documents with overlapping parts
        if(len(sum_text.split()) > 500):
            text_parts = get_split(sum_text, 500, 50)
            tensors = tokenizer(
                text_parts, padding="max_length", truncation="only_first")
            encodings.append(tensors)
            labels.append(news_dataframe.iloc[idx]['label'])
        else:
            encodings.append(
                tokenizer(sum_text, padding="max_length", truncation="only_first"))
            labels.append(news_dataframe.iloc[idx]['label'])

        if is_use_tweets:
            tweet_texts = get_tweet_texts(root_path)

            if(len(tweet_texts.split()) > 500):
                text_parts = get_split(tweet_texts, 500, 0)
                tensors = tokenizer(
                    text_parts, padding="max_length", truncation="only_first")
                # Dimensional mean of the tensor to represent all parts of the text
                mean_input_ids = list(np.mean(tensors.input_ids, axis=0))
                mean_attention_mask = list(np.mean(tensors.attention_mask, axis=0))
                tensors.data['input_ids'] = mean_input_ids
                tensors.data['attention_mask'] = mean_attention_mask
                tweet_embeddings.append(tensors)
            else:
                tweet_embeddings.append(
                    tokenizer(tweet_texts, padding="max_length", truncation="only_first"))

    retweet_count_matrix = None
    if is_use_retweets:
        # This needs to be a matrix of the right shape
        retweet_count_matrix = np.zeros((len(retweet_counts), 1))
        for i in range(len(retweet_counts)):
            retweet_count_matrix[i][0] = retweet_counts[i]

    source_embeddings = None
    if is_use_sources:
        source_embeddings = tokenizer(
            sources, padding="max_length", truncation="only_first")
    
    author_embeddings = None
    if is_use_authors:
        author_embeddings = tokenizer(
            authors, padding="max_length", truncation="only_first")

    return encodings, labels, source_embeddings, author_embeddings, tweet_embeddings, retweet_count_matrix


def get_tweet_texts(root_path):
    tweet_texts = []
    for path in glob.glob(root_path + "/tweets/*"):
        with open(path) as f:
            data = json.load(f)
            tweet_texts.append(
                re.sub(r'http\S+', '', data['text'], flags=re.MULTILINE))
    if not tweet_texts:
        tweet_texts.append("")
    # Remove duplicate values
    tweet_texts = list(dict.fromkeys(tweet_texts))
    tweet_text = ". ".join(tweet_texts)
    return tweet_text


def get_tweet_authors(root_path):
    tweet_authors = []
    for path in glob.glob(root_path + "/tweets/*"):
        with open(path) as f:
            data = json.load(f)
            tweet_authors.append(data['user']['name'])
    if not tweet_authors:
        tweet_authors.append("")
    # Remove duplicate values
    tweet_authors = list(dict.fromkeys(tweet_authors))
    authors = ",".join(tweet_authors)
    return authors


def get_retweet_count(root_path):
    retweet_count = 0
    for path in glob.glob(root_path + "/retweets/*"):
        with open(path) as f:
            data = json.load(f)
            retweet_count = retweet_count + len(data['retweets'])
    return retweet_count


def get_news_content_encodings_truncated(root_dir, news_dataframe, tokenizer, summary=NO_SUMMARY):
    texts = []
    encodings = []
    labels = list(news_dataframe['label'])
    for idx in range(len(news_dataframe)):
        path = ""
        if news_dataframe.iloc[idx]['label'] == 0:
            path = os.path.join(root_dir, news_dataframe.iloc[idx]['type'],
                                'real', news_dataframe.iloc[idx, 0], "news content.json")
        else:
            path = os.path.join(root_dir, news_dataframe.iloc[idx]['type'],
                                'fake', news_dataframe.iloc[idx, 0], "news content.json")
        if(summary == ABSTRACTIVE_SUMMARY):
            path = path[:-5] + "_summary_abstractive.json"
        if(summary == EXTRACTIVE_SUMMARY):
            path = path[:-5] + "_summary_extractive.json"

        with open(path) as f:
            data = json.load(f)
        with open(path) as f:
            data = json.load(f)
        sum_text = data['title'] + ". " + data['text']
        texts.append(sum_text)
    # It is important to tokenize all texts at once to ensure a consistent shape
    encodings = tokenizer(texts, padding=True, truncation=True)
    return encodings, labels


def write_results_to_file(evaluation_results, train_args, train_dataset, test_dataset, is_use_summary_train, is_use_summary_test, is_use_text, is_use_sources, is_use_authors, is_use_tweets, is_use_retweets, is_truncated, is_ensemble, random_state, is_hp_search):
    evaluation_results['epoch'] = train_args.num_train_epochs
    evaluation_results['timestamp'] = datetime.now()
    evaluation_results['train_dataset'] = train_dataset
    evaluation_results['test_dataset'] = test_dataset
    evaluation_results['text_type_train'] = is_use_summary_train
    evaluation_results['text_type_test'] = is_use_summary_test
    evaluation_results['is_use_text'] = is_use_text
    evaluation_results['is_use_sources'] = is_use_sources
    evaluation_results['is_use_authors'] = is_use_authors
    evaluation_results['is_use_tweets'] = is_use_tweets
    evaluation_results['is_use_retweets'] = is_use_retweets
    evaluation_results['is_truncated'] = is_truncated
    evaluation_results['is_ensemble'] = is_ensemble
    evaluation_results['random_state'] = random_state
    evaluation_results['is_hp_search'] = is_hp_search
    evaluation_results['learning_rate'] = train_args.learning_rate
    evaluation_results['train_batch_size'] = train_args.train_batch_size
    evaluation_results['eval_batch_size'] = train_args.per_device_eval_batch_size
    evaluation_results['gradient_accumulation_steps'] = train_args.gradient_accumulation_steps
    evaluation_results['weight_decay'] = train_args.weight_decay

    df = pd.DataFrame(evaluation_results, index=[7])
    if(os.path.isfile("results.csv")):
        df.to_csv("results.csv", mode="a+", index=False, header=False)
    else:
        df.to_csv("results.csv", mode="a+", index=False, header=True)

def generate_datasets(train_set, val_set, test_set, tokenizer, text_type_train, text_type_val, is_truncated, is_use_sources, is_use_authors, is_use_tweets, is_use_retweets):
    if is_truncated:
        train_encodings, train_labels = get_news_content_encodings_truncated(
            PATH_TO_DATASET, train_set, tokenizer, text_type_train)
        val_encodings, val_labels = get_news_content_encodings_truncated(
            PATH_TO_DATASET, val_set, tokenizer, text_type_val)

        train_dataset = FakeNewsNetDatasetTruncated(
            train_encodings, train_labels)
        val_dataset = FakeNewsNetDatasetTruncated(val_encodings, val_labels)

        if test_set is not None:
            test_encodings, test_labels = get_news_content_encodings_truncated(
                PATH_TO_DATASET, test_set, tokenizer, text_type_val)
            test_dataset = FakeNewsNetDatasetTruncated(
                test_encodings, test_labels)

    elif is_use_sources or is_use_authors or is_use_tweets or is_use_retweets:
        train_encodings, train_labels, train_source_embeddings, train_author_embeddings, train_tweet_embeddings, train_retweet_count = get_news_content_encodings(
            PATH_TO_DATASET, train_set, tokenizer, text_type_train, is_use_sources, is_use_authors, is_use_tweets, is_use_retweets)
        val_encodings, val_labels, val_source_embeddings, val_author_embeddings, val_tweet_embeddings, val_retweet_count = get_news_content_encodings(
            PATH_TO_DATASET, val_set, tokenizer, text_type_val, is_use_sources, is_use_authors, is_use_tweets, is_use_retweets)

        train_dataset = FakeNewsNetDatasetMetadata(train_encodings, train_labels,
                                                 train_source_embeddings, train_author_embeddings, train_tweet_embeddings, train_retweet_count)
        val_dataset = FakeNewsNetDatasetMetadata(
            val_encodings, val_labels, val_source_embeddings, val_author_embeddings, val_tweet_embeddings, val_retweet_count)

        if test_set is not None:
            test_encodings, test_labels, test_source_embeddings, test_author_embeddings, test_tweet_embeddings, test_retweet_count = get_news_content_encodings(
                PATH_TO_DATASET, test_set, tokenizer, text_type_val)
            test_dataset = FakeNewsNetDatasetMetadata(
                test_encodings, test_labels, test_source_embeddings, test_author_embeddings, test_tweet_embeddings, test_retweet_count)
    else:
        train_encodings, train_labels, _, _, _, _ = get_news_content_encodings(
            PATH_TO_DATASET, train_set, tokenizer, text_type_train, False, False, False, False)
        val_encodings, val_labels, _, _, _, _ = get_news_content_encodings(
            PATH_TO_DATASET, val_set, tokenizer, text_type_val, False, False, False, False)

        train_dataset = FakeNewsNetDataset(train_encodings, train_labels)
        val_dataset = FakeNewsNetDataset(val_encodings, val_labels)

        if test_set is not None:
            test_encodings, test_labels, _, _, _ = get_news_content_encodings(
                PATH_TO_DATASET, test_set, tokenizer, text_type_val, False, False, False, False)
            test_dataset = FakeNewsNetDataset(test_encodings, test_labels, False, False, False)

    if test_set is None:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset
