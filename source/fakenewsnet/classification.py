import numpy
import torch
from sklearn.metrics import (accuracy_score, log_loss,
                             precision_recall_fscore_support)
from torch import nn
from transformers import (BertForSequenceClassification, BertModel)

# Basically a custom implementation of BertForSequenceClassification with the possibility to freeze the layers or the embeddings
class BertClassifier(nn.Module):
    # This parameters need to be equal to the used BERT model
    def __init__(self, bert_model_path=None, labels_count=2, hidden_dim=4*768, dropout=0.1, freeze_emb=False, freeze_all=False):
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

    def forward(self, attention_mask, input_ids, labels, token_type_ids=None, input_ids_1=None, input_ids_2=None, input_ids_3=None, attention_mask_1=None, attention_mask_2=None, attention_mask_3=None):
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
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, self.config['labels_count']), labels.view(-1))

        output = {}

        output['loss'] = loss
        output['logits'] = logits

        return output

class BertBaseClassifier(nn.Module):
    # This parameters need to be equal to the used BERT model
    def __init__(self, bert_model_path=None, labels_count=2, hidden_dim=768, dropout=0.1, freeze_emb=False, freeze_all=False):
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

    def forward(self, attention_mask, input_ids, labels, token_type_ids=None):
        
        hidden_state = self.bert(input_ids, attention_mask)[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)

        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, self.config['labels_count']), labels.view(-1))

        output = {}

        output['loss'] = loss
        output['logits'] = logits

        return output

class BertExtrasClassifier(nn.Module):
    def __init__(self, is_use_text, is_use_sources, is_use_authors, is_use_tweets, is_use_retweets, labels_count=2, dropout=0.1, freeze_emb=False, freeze_all=False):
        super().__init__()

        input_data = [is_use_text, is_use_text, is_use_text, is_use_text, is_use_sources, is_use_authors, is_use_tweets]
        hidden_dim = 0

        for data in input_data:
            if data:
                hidden_dim += 768

        if is_use_retweets:
            hidden_dim += 1
        
        self.config = {
            'labels_count': labels_count,
            'is_use_text': is_use_text,
            'is_use_sources': is_use_sources,
            'is_use_authors': is_use_authors,
            'is_use_tweets': is_use_tweets,
            'is_use_retweets': is_use_retweets
        }

        self.bert_text = BertModel.from_pretrained("bert-base-uncased")

        self.bert_metadata = BertModel.from_pretrained("bert-base-uncased")

        # The second BERT model should only produce embeddings without actually finetuning it
        for param in self.bert_metadata.parameters():
            param.requires_grad = False

        if(freeze_all):
            # Freeze all layers -> BERT baseline without any training (train only classifier)
            for param in self.bert_text.parameters():
                param.requires_grad = False

        if(freeze_emb):
            # Freeze embeddings only
            for param in self.bert_text.embeddings.parameters():
                param.requires_grad = False

        self.pre_classifier = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, labels_count)
        self.dropout = nn.Dropout(dropout)

    def forward(self, attention_mask, input_ids, labels, attention_mask_source=None, input_ids_source=None, attention_mask_author=None, input_ids_author=None, attention_mask_tweet=None, input_ids_tweet=None, retweet_count=None, input_ids_1=None, input_ids_2=None, input_ids_3=None, attention_mask_1=None, attention_mask_2=None, attention_mask_3=None):
        tensors = []

        if self.config['is_use_text']:
            hidden_state = self.bert_text(input_ids, attention_mask)[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            tensors.append(pooled_output)

            hidden_state = self.bert_text(input_ids_1, attention_mask_1)[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            tensors.append(pooled_output)

            hidden_state = self.bert_text(input_ids_2, attention_mask_2)[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            tensors.append(pooled_output)

            hidden_state = self.bert_text(input_ids_3, attention_mask_3)[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            tensors.append(pooled_output)


        if self.config['is_use_sources']:
            hidden_state_source = self.bert_metadata(
                input_ids_source, attention_mask=attention_mask_source)[0]  # (bs, seq_len, dim)
            pooled_output_source = hidden_state_source[:, 0]  # (bs, dim)
            tensors.append(pooled_output_source)

        if self.config['is_use_authors']:
            hidden_state_author = self.bert_metadata(
                input_ids_author, attention_mask=attention_mask_author)[0]  # (bs, seq_len, dim)
            pooled_output_author = hidden_state_author[:, 0]  # (bs, dim)
            tensors.append(pooled_output_author)

        if self.config['is_use_tweets']:
            hidden_state_tweet = self.bert_metadata(
                input_ids_tweet, attention_mask=attention_mask_tweet)[0]  # (bs, seq_len, dim)
            pooled_output_tweet = hidden_state_tweet[:, 0]  # (bs, dim)
            tensors.append(pooled_output_tweet)
        
        if self.config['is_use_retweets']:
            tensors.append(retweet_count.float())

        # Concat representations
        pooled_output = torch.cat(tensors, dim=1)

        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, self.config['labels_count']), labels.view(-1))

        output = {}

        output['loss'] = loss
        output['logits'] = logits

        return output

class FakeNewsNetDataset(torch.utils.data.Dataset):
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
                item['input_ids_' + str(i)] = numpy.zeros(512, dtype=int)
                item['attention_mask_' + str(i)] = numpy.zeros(512, dtype=int)
        
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class FakeNewsNetDatasetMetadata(FakeNewsNetDataset):
    def __init__(self, encodings, labels, source_embeddings=None, author_embeddings=None, tweet_embeddings=None, retweet_counts=None):
        self.encodings = encodings
        self.labels = labels
        self.source_embeddings = source_embeddings
        self.author_embeddings = author_embeddings
        self.tweet_embeddings = tweet_embeddings
        self.retweet_counts = retweet_counts

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        if self.source_embeddings:
            for key, val in self.source_embeddings.items():
                # Only used when training on sequence pairs disregard here
                if(key == "token_type_ids"):
                    continue
                item[key + "_source"] = val[idx]
        
        if self.author_embeddings:
            for key, val in self.author_embeddings.items():
                if(key == "token_type_ids"):
                    continue
                item[key + "_author"] = val[idx]

        if self.tweet_embeddings:
            tmp_list = []
            for key in self.tweet_embeddings[idx]['input_ids']:
                tmp_list.append(int(round(key)))

            item['input_ids' + "_tweet"] = tmp_list
            tmp_list2 = []
            for key in self.tweet_embeddings[idx]['attention_mask']:
                tmp_list2.append(key)

            item['attention_mask' + "_tweet"] = tmp_list2
        
        if self.retweet_counts is not None:
            item['retweet_count'] = self.retweet_counts[idx]

        return item


class FakeNewsNetDatasetTruncated(FakeNewsNetDataset):
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def init_hierarchical_model():
    return BertClassifier()

def init_full_text_model():
    return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


def init_metadata_model(is_use_text=True, is_use_sources=True, is_use_authors=True, is_use_tweets=True, is_use_retweets=True):
    return BertExtrasClassifier(is_use_text, is_use_sources, is_use_authors, is_use_tweets, is_use_retweets)


def init_emb_model():
    return BertBaseClassifier(freeze_emb=True, freeze_all=False)


def init_base_model():
    return BertBaseClassifier(freeze_emb=False, freeze_all=True, labels_count=2)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }



def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 3),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4]),
    }


def hp_objective(metrics):
    return metrics['eval_f1']


def hyperparameter_search(trainer, trials=5):
    best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize",
                                             hp_space=hp_space, compute_objective=hp_objective)
    print(best_run)
    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)
