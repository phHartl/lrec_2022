import numpy as np
import torch
import gc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from summarizer import Summarizer
from summarizer.coreference_handler import CoreferenceHandler
from transformers import (BartForConditionalGeneration, BartTokenizerFast,
                          DistilBertConfig,
                          DistilBertModel, DistilBertTokenizerFast, BertModel)
from torch import nn

sum_tokenizer = BartTokenizerFast.from_pretrained(
    'sshleifer/distilbart-cnn-12-6')
sum_model = BartForConditionalGeneration.from_pretrained(
    'sshleifer/distilbart-cnn-12-6')

custom_config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
custom_config.output_hidden_states = True
custom_tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased")
custom_model = DistilBertModel.from_pretrained(
    "distilbert-base-uncased", config=custom_config)
handler = CoreferenceHandler("en_core_web_sm")


def convert_to_int(rating):
    if(rating == 'TRUE' or rating == "true" or rating == True):
        return 0
    if(rating == 'FALSE' or rating == "false" or rating == False):
        return 1
    if(rating == "partially false"):
        return 2
    else:
        return 3


def convert_to_rating(int):
    if(int == 0):
        return "true"
    if(int == 1):
        return "false"
    if(int == 2):
        return "partially false"
    else:
        return "other"


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def get_extractive_summary(extractive_sum_model, text):
    sum_text = extractive_sum_model(text, ratio=0.4)
    if not sum_text:
        return text
    else:
        return sum_text


def generate_extractive_summaries(dataframe, filename):
    extractive_sum_model = Summarizer(
        custom_model=custom_model, custom_tokenizer=custom_tokenizer, handler=handler, random_state=43)

    dataframe['text_extractive'] = dataframe.apply(
        lambda x: get_extractive_summary(extractive_sum_model, x['text']), axis=1)
    dataframe = dataframe[["public_id", "title",
                           "text", "text_extractive", "our rating"]]
    dataframe.to_csv(filename, index=False)

def refresh_cuda_memory():
    """
    Re-allocate all cuda memory to help alleviate fragmentation
    """
    # Run a full garbage collect first so any dangling tensors are released
    gc.collect()

    # Then move all tensors to the CPU
    locations = {}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue

        locations[obj] = obj.device
        obj.data = obj.data.cpu()
        if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
            obj.grad.data = obj.grad.cpu()

    # Now empty the cache to flush the allocator
    torch.cuda.empty_cache()

    # Finally move the tensors back to their associated GPUs
    for tensor, device in locations.items():
        tensor.data = tensor.to(device)
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.to(device)


def get_abstractive_summary(text, sum_tokenizer, sum_model):
    summary = ""
    oom = False
    if(len(text.split()) > 1000):
        texts = get_split(text, 1000, 0)
        inputs = sum_tokenizer(texts, return_tensors='pt', truncation="only_first", padding="max_length", max_length=1024).input_ids
        # Generate Summary
        try:
            output = sum_model.generate(inputs.cuda(), min_length = 400, max_length = 512, top_k=100, top_p=.95, do_sample=True)
        except RuntimeError:
            oom = True
            print("OOM Error")
        
        if oom:
            output = sum_model.cpu().generate(inputs.cpu(), min_length = 400, max_length = 512, top_k=100, top_p=.95, do_sample=True)
            print("Iteration ran on cpu instead")
            sum_model.cuda()
            refresh_cuda_memory()

        sum_texts = sum_tokenizer.batch_decode(output, skip_special_tokens=True)
        summary = "".join(sum_texts)
    else:
        inputs = sum_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=1024).input_ids
        # Generate Summary
        try:
            output = sum_model.generate(inputs.cuda(), min_length = int(len(text.split())*0.4), max_length = 512, top_k=100, top_p=.95, do_sample=True)
        except RuntimeError:
            oom = True
            print("OOM Error")
        if oom:
            output = sum_model.cpu().generate(inputs.cpu(), min_length = int(len(text.split())*0.4), max_length = 512, top_k=100, top_p=.95, do_sample=True)
            sum_model.cuda()
            refresh_cuda_memory()

        sum_texts = sum_tokenizer.batch_decode(output, skip_special_tokens=True)
        summary = sum_texts[0]

    return summary

def generate_abstractive_summaries(dataframe, filename):
    dataframe['text_abstractive'] = dataframe.apply(
        lambda x: get_abstractive_summary(x['text'], sum_tokenizer, sum_model), axis=1)
    dataframe = dataframe[["public_id", "title", "text",
                           "text_extractive", "text_abstractive", "our rating"]]
    dataframe.to_csv(filename, index=False)


def get_encodings_test(dataframe, tokenizer, summary=0):
    encodings = []
    for idx in range(len(dataframe)):
        if(summary == 2):
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text_extractive']
        if(summary == 1):
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text_abstractive']
        else:
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text']
        # Split longer texts into documents with overlapping parts
        if(len(sum_text.split()) > 500):
            text_parts = get_split(sum_text, 500, 50)
            tensors = tokenizer(
                text_parts, padding="max_length", truncation="only_first")
            encodings.append(tensors)
        else:
            encodings.append(
                tokenizer(sum_text, padding="max_length", truncation="only_first"))

    return encodings


def get_encodings(dataframe, tokenizer, summary=0):
    encodings = []
    labels = []
    for idx in range(len(dataframe)):
        if(summary == 2):
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text_extractive']
        if(summary == 1):
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text_abstractive']
        elif(summary == 0):
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text']

        # Split longer texts into documents with overlapping parts
        if(len(sum_text.split()) > 500):
            text_parts = get_split(sum_text, 500, 50)
            tensors = tokenizer(
                text_parts, padding="max_length", truncation="only_first")
            encodings.append(tensors)
            labels.append(dataframe.iloc[idx]['label'])
        else:
            encodings.append(
                tokenizer(sum_text, padding="max_length", truncation="only_first"))
            labels.append(dataframe.iloc[idx]['label'])


    return encodings, labels

# This class represent one part of the dataset (either training, validation or test via subclass)
class CheckThatLabDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        internal_counter = 0
        if type(self.encodings[idx]['input_ids'][0]) == list:
            for encoding in self.encodings[idx]['input_ids']:
                if internal_counter < 4:
                    if internal_counter != 0:
                        item['input_ids_' + str(internal_counter)] = encoding
                    else:
                        item['input_ids'] = encoding
                internal_counter += 1
        else:
            item['input_ids'] = self.encodings[idx]['input_ids']

        internal_counter = 0
        if type(self.encodings[idx]['attention_mask'][0]) == list:
            for encoding in self.encodings[idx]['attention_mask']:
                if internal_counter < 4:
                    if internal_counter != 0:
                        item['attention_mask_' + str(internal_counter)] = encoding
                    else:
                        item['attention_mask'] = encoding
                internal_counter += 1
        else:
            item['attention_mask'] = self.encodings[idx]['attention_mask']

        for i in range(1,4):
            if not 'input_ids_' + str(i) in item:
                item['input_ids_' + str(i)] = np.zeros(512, dtype=int)
                item['attention_mask_' + str(i)] = np.zeros(512, dtype=int)
        
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class CheckThatLabDatasetTest(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {}
        internal_counter = 0
        if type(self.encodings[idx]['input_ids'][0]) == list:
            for encoding in self.encodings[idx]['input_ids']:
                if internal_counter < 4:
                    if internal_counter != 0:
                        item['input_ids_' + str(internal_counter)] = encoding
                    else:
                        item['input_ids'] = encoding
                internal_counter += 1
        else:
            item['input_ids'] = self.encodings[idx]['input_ids']

        internal_counter = 0
        if type(self.encodings[idx]['attention_mask'][0]) == list:
            for encoding in self.encodings[idx]['attention_mask']:
                if internal_counter < 4:
                    if internal_counter != 0:
                        item['attention_mask_' + str(internal_counter)] = encoding
                    else:
                        item['attention_mask'] = encoding
                internal_counter += 1
        else:
            item['attention_mask'] = self.encodings[idx]['attention_mask']

        for i in range(1,4):
            if not 'input_ids_' + str(i) in item:
                item['input_ids_' + str(i)] = np.zeros(512, dtype=int)
                item['attention_mask_' + str(i)] = np.zeros(512, dtype=int)

        return item

    def __len__(self):
        return len(self.encodings)

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


# Basically a custom implementation of BertForSequenceClassification with the possibility to freeze the layers or the embeddings
class BertClassifier(nn.Module):
    # This parameters need to be equal to the used BERT model
    def __init__(self, bert_model_path=None, labels_count=4, hidden_dim=4*768, dropout=0.1, freeze_emb=False, freeze_all=False):
        super().__init__()

        self.config = {
            'labels_count': labels_count,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
        }

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if(freeze_all):
            # Freeze all layers -> BERT baseline without any training (train only classifier)
            for param in self.bert.parameters():
                param.requires_grad = False

        if(freeze_emb):
            # Freeze embeddings only
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.pre_classifier = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, labels_count)
        self.dropout = nn.Dropout(dropout)

    def forward(self, attention_mask, input_ids, labels=None, token_type_ids=None, input_ids_1=None, input_ids_2=None, input_ids_3=None, attention_mask_1=None, attention_mask_2=None, attention_mask_3=None):
        tensors = []
        
        hidden_state = self.bert(input_ids, attention_mask)[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        tensors.append(pooled_output)

        hidden_state = self.bert(input_ids_1, attention_mask_1)[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        tensors.append(pooled_output)

        hidden_state = self.bert(input_ids_2, attention_mask_2)[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        tensors.append(pooled_output)

        hidden_state = self.bert(input_ids_3, attention_mask_3)[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        tensors.append(pooled_output)

        pooled_output = torch.cat(tensors, dim=1)

        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config['labels_count']), labels.view(-1))

        output = {}

        if loss is not None:
            output['loss'] = loss
        
        output['logits'] = logits

        return output

def init_full_text_model():
    return BertClassifier()
    

def init_frozen_model():
    return BertClassifier(freeze_all=True)
