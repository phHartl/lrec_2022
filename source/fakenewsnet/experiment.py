# %%
import argparse
from copy import copy
from statistics import mean, mode

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, Trainer, TrainingArguments

from classification import *
from utils import (ABSTRACTIVE_SUMMARY, EXTRACTIVE_SUMMARY,
                   NO_SUMMARY, PATH_TO_DATASET_CSV, POLITIFACT, GOSSIPCOP,
                   check_if_file_exists, generate_datasets, prep_dataframe,
                   write_results_to_file)

# %% Define experiment parameters from command line or use default values
parser = argparse.ArgumentParser(description="Run an experiment iteration")

parser.add_argument('--train_dataset', dest='TRAIN_DATASET', default=GOSSIPCOP)
parser.add_argument('--validation_dataset', dest='VALIDATION_DATASET', default=POLITIFACT)
parser.add_argument('--test_dataset', dest='TEST_DATASET', default=GOSSIPCOP)

parser.add_argument('--text_type_train', dest='TEXT_TYPE_TRAIN', default=NO_SUMMARY)
parser.add_argument('--text_type_test', dest='TEXT_TYPE_TEST', default=NO_SUMMARY)

parser.add_argument('--no_text', dest='IS_USE_TEXT', action='store_const', const=False, default=True)
parser.add_argument('--sources', dest='IS_USE_SOURCES', action='store_const', const=True, default=False)
parser.add_argument('--authors', dest='IS_USE_AUTHORS', action='store_const', const=True, default=False)
parser.add_argument('--tweets', dest='IS_USE_TWEETS', action='store_const', const=True, default=False)
parser.add_argument('--retweets', dest='IS_USE_RETWEETS', action='store_const', const=True, default=False)

parser.add_argument('--truncate', dest='IS_TRUNCATED', action='store_const', const=True, default=False)
parser.add_argument('--ensemble', dest='IS_ENSEMBLE', action='store_const', const=True, default=True)

parser.add_argument('--random_state', dest='RANDOM_STATE', type=int, default=50)

parser.add_argument('--hyperparameter_search', dest='IS_HP_SEARCH', action='store_const', const=True, default=False)
parser.add_argument('--train_val_test_split', dest='IS_TRAIN_VAL_TEST_SPLIT', action='store_const', const=True, default=False)
parser.add_argument('--baseline', dest='IS_BASELINE', action="store_const", const=True, default=False)

args, unknown = parser.parse_known_args()

args = vars(args)
# %% Print the current configuration
print("Running experiment with following configuration:")
print("Train dataset " + args['TRAIN_DATASET'])
print("Validation dataset " + args['VALIDATION_DATASET'])
if(args['IS_TRAIN_VAL_TEST_SPLIT']):
    print("Test dataset " + args['TEST_DATASET'])
if(args['IS_ENSEMBLE']):
    print("Using ensemble classifier")
else:
    print("Using " + args['TEXT_TYPE_TRAIN'] + " text for training")
    print("Using " + args['TEXT_TYPE_TEST'] + " text for validation/test")
if(args['IS_BASELINE']):
    args['IS_TRUNCATED'] = True
    print("Running baseline configuration")
if(args['IS_TRUNCATED']):
    print("All texts are truncated to 512 word pieces")
if(args['IS_USE_SOURCES'] or args['IS_USE_AUTHORS'] or args['IS_USE_TWEETS'] or args['IS_USE_RETWEETS']):
    print("Additional metadata is used: ")
    if(args['IS_USE_SOURCES']):
        print("News source urls")
    if(args['IS_USE_AUTHORS']):
        print("News & related tweet authors")
    if(args['IS_USE_TWEETS']):
        print("Related tweet texts")
    if(args['IS_USE_RETWEETS']):
        print("Amount of retweets")
if(args['IS_HP_SEARCH']):
    print("We perform a hyperparameter search with ten iterations")
print("The current seed is " + str(args['RANDOM_STATE']))
print("We are using " + str(torch.cuda.device_count()) + " GPU(s)")
# %%
# Prepare dataframes
df_politifact = prep_dataframe(PATH_TO_DATASET_CSV + POLITIFACT)
df_gossipcop = prep_dataframe(PATH_TO_DATASET_CSV + GOSSIPCOP)

# %% Remove all articles without any text (summary tag not useful here atm because some articles got no text)
df_politifact['file'] = df_politifact.apply(lambda x: check_if_file_exists(
    x['id'], x['label'], x['type']), axis=1)
df_politifact = df_politifact[df_politifact['file'] == True]

df_gossipcop['file'] = df_gossipcop.apply(lambda x: check_if_file_exists(
    x['id'], x['label'], x['type']), axis=1)
df_gossipcop = df_gossipcop[df_gossipcop['file'] == True]

# %% Split dataset into train, validation and test set while keeping distribution

# Split into train and test set
df_politifact_train, df_politifact_test = train_test_split(
    df_politifact, test_size=0.2, random_state=args['RANDOM_STATE'], stratify=df_politifact['label'])
df_gossipcop_train, df_gossipcop_test = train_test_split(
    df_gossipcop, test_size=0.2, random_state=args['RANDOM_STATE'], stratify=df_gossipcop['label'])

# Split train set into validation and train set if necessary
if args['IS_HP_SEARCH'] or args['IS_TRAIN_VAL_TEST_SPLIT']:
    df_politifact_train, df_politifact_val = train_test_split(
        df_politifact_train, test_size=0.25, random_state=args['RANDOM_STATE'], stratify=df_politifact_train['label'])
    df_gossipcop_train, df_gossipcop_val = train_test_split(
        df_gossipcop_train, test_size=0.25, random_state=args['RANDOM_STATE'], stratify=df_gossipcop_train['label'])
# Else use the first split as train & test data
else:
    df_politifact_val = copy(df_politifact_test)
    df_politifact_test = None
    df_gossipcop_val = copy(df_gossipcop_test)
    df_gossipcop_test = None

# When using separate datasets use the whole dataset
if args['TRAIN_DATASET'] != args['VALIDATION_DATASET']:
    if args['TRAIN_DATASET'] == POLITIFACT:
        df_politifact_train = df_politifact
        df_politifact_val = df_gossipcop
    else:
        df_gossipcop_train = df_gossipcop
        df_gossipcop_val = df_politifact
# %%
# Oversample minority classes to ensure a better training -> done for both cases to ensure better comparability
ros = RandomOverSampler(random_state=args['RANDOM_STATE'])
x_resampled, y_resampled = ros.fit_resample(df_gossipcop_train[[
                                            "id", "news_url", "title", "tweet_ids", "type", "file"]], df_gossipcop_train["label"])
data_oversampled = pd.concat(
    [pd.DataFrame(x_resampled), pd.DataFrame(y_resampled)], axis=1)
data_oversampled.head()
df_gossipcop_train = data_oversampled

ros = RandomOverSampler(random_state=args['RANDOM_STATE'])
x_resampled, y_resampled = ros.fit_resample(df_politifact_train[[
                                            "id", "news_url", "title", "tweet_ids", "type", "file"]], df_politifact_train["label"])
data_oversampled = pd.concat(
    [pd.DataFrame(x_resampled), pd.DataFrame(y_resampled)], axis=1)
data_oversampled.head()
df_politifact_train = data_oversampled
# %% Generate datasets
if(args['TRAIN_DATASET'] == POLITIFACT):
    train_set = pd.concat([df_politifact_train])
elif(args['TRAIN_DATASET'] == GOSSIPCOP):
    train_set = pd.concat([df_gossipcop_train])

if(args['VALIDATION_DATASET'] == POLITIFACT):
    val_set = pd.concat([df_politifact_val])
elif(args['VALIDATION_DATASET'] == GOSSIPCOP):
    val_set = pd.concat([df_gossipcop_val])

if(args['TEST_DATASET'] == POLITIFACT and df_politifact_test is not None):
    test_set = pd.concat([df_politifact_test])
elif(args['TEST_DATASET'] == GOSSIPCOP and df_gossipcop_test is not None):
    test_set = pd.concat([df_gossipcop_test])
else:
    test_set = None

# %% Get encodings for each scenario
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

if args['IS_ENSEMBLE']:
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for i in [NO_SUMMARY, ABSTRACTIVE_SUMMARY, EXTRACTIVE_SUMMARY]:
        train_dataset, val_dataset, test_dataset = generate_datasets(
            train_set, val_set, test_set, tokenizer, i, i, args['IS_TRUNCATED'], args['IS_USE_SOURCES'], args['IS_USE_AUTHORS'], args['IS_USE_TWEETS'], args['IS_USE_RETWEETS'])
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)
else:
    train_dataset, val_dataset, test_dataset = generate_datasets(
        train_set, val_set, test_set, tokenizer, args['TEXT_TYPE_TRAIN'], args['TEXT_TYPE_TEST'], args['IS_TRUNCATED'], args['IS_USE_SOURCES'], args['IS_USE_AUTHORS'], args['IS_USE_TWEETS'], args['IS_USE_RETWEETS'])
# %% Set training variables
if args['IS_TRAIN_VAL_TEST_SPLIT']:
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        # num_train_epochs=3,              # total number of training epochs
        evaluation_strategy="steps",
        # batch size per device during training
        per_device_train_batch_size=round(8/torch.cuda.device_count()),
        per_device_eval_batch_size=8,    # batch size for evaluation
        warmup_ratio=0.1,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        # greater_is_better=True,
        seed=args['RANDOM_STATE'],
    )
else:
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        # batch size per device during training
        per_device_train_batch_size=round(8/torch.cuda.device_count()),
        per_device_eval_batch_size=8,    # batch size for evaluation
        warmup_ratio=0.1,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args['RANDOM_STATE'],
    )

# %% Get trainer instances
if not args['IS_ENSEMBLE']:
    if(args['IS_TRUNCATED']):
        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model_init=init_full_text_model,
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,
            data_collator=None,            # evaluation dataset
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    elif(args['IS_USE_SOURCES'] or args['IS_USE_AUTHORS'] or args['IS_USE_TWEETS'] or args['IS_USE_RETWEETS']):
        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=init_metadata_model(
                args['IS_USE_TEXT'], args['IS_USE_SOURCES'], args['IS_USE_AUTHORS'], args['IS_USE_TWEETS'], args['IS_USE_RETWEETS']),
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,
            data_collator=None,            # evaluation dataset
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=init_hierarchical_model(),
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,
            data_collator=None,            # evaluation dataset
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    if(args['IS_BASELINE']):
        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=init_base_model(),
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,
            data_collator=None,            # evaluation dataset
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
else:
    def get_trainer(index):
        if args['IS_TRUNCATED']:
            trainer = Trainer(
                # the instantiated 🤗 Transformers model to be trained
                model_init=init_full_text_model,
                args=training_args,                  # training arguments, defined above
                train_dataset=train_datasets[index],         # training dataset
                eval_dataset=val_datasets[index],
                data_collator=None,            # evaluation dataset
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )

        elif(args['IS_USE_SOURCES'] or args['IS_USE_AUTHORS'] or args['IS_USE_TWEETS'] or args['IS_USE_RETWEETS']):
            trainer = Trainer(
                # the instantiated 🤗 Transformers model to be trained
                model=init_metadata_model(
                    args['IS_USE_TEXT'], args['IS_USE_SOURCES'], args['IS_USE_AUTHORS'], args['IS_USE_TWEETS'], args['IS_USE_RETWEETS']),
                args=training_args,                  # training arguments, defined above
                train_dataset=train_datasets[index],         # training dataset
                eval_dataset=val_datasets[index],
                data_collator=None,            # evaluation dataset
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
        else:
            trainer = Trainer(
                # the instantiated 🤗 Transformers model to be trained
                model_init=init_hierarchical_model,
                args=training_args,                  # training arguments, defined above
                train_dataset=train_datasets[index],         # training dataset
                eval_dataset=val_datasets[index],
                data_collator=None,            # evaluation dataset
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
        return trainer

# %% Train model
if not args['IS_ENSEMBLE']:
    if args['IS_HP_SEARCH']:
        hyperparameter_search(trainer)
    else:
        trainer.train()
else:
    prediction_outputs = []
    for j in range(len(train_datasets)):
        trainer = get_trainer(j)
        trainer.train()
        if args['IS_TRAIN_VAL_TEST_SPLIT']:
            prediction_outputs.append(trainer.predict(test_datasets[j]))
        else:
            prediction_outputs.append(trainer.predict(val_datasets[j]))
        # remove trainer from cuda memory
        trainer_args = trainer.args
        trainer = None
# %% Print evaluation results
if not (args['IS_ENSEMBLE'] or args['IS_TRAIN_VAL_TEST_SPLIT']):
    evaluation_results = trainer.evaluate()
    del evaluation_results['eval_runtime']
    del evaluation_results['eval_samples_per_second']
    del evaluation_results['eval_steps_per_second']
    print(evaluation_results)
elif args['IS_ENSEMBLE']:
    # This probably should be a matrix with one row being all three predictions? -> numpy
    preds = []
    eval_losses = []
    labels = prediction_outputs[0].label_ids
    for prediction in prediction_outputs:
        preds.append(prediction.predictions.argmax(-1))
        eval_losses.append(prediction.metrics['test_loss'])

    predictions = []
    for k in range(len(preds[0])):
        pred_base = preds[0][k]
        pred_abstractive = preds[1][k]
        pred_extractive = preds[2][k]
        predictions.append(
            mode([pred_base, pred_abstractive, pred_extractive]))

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary")

    acc = accuracy_score(labels, predictions)

    test_results = {
        'eval_loss': mean(eval_losses),
        'eval_accuracy': acc,
        'eval_f1': f1,
        'eval_precision': precision,
        'eval_recall': recall
    }
# %% Print test set results (useful when using hyperparameter search)
if args['IS_ENSEMBLE']:
    print(test_results)
    write_results_to_file(test_results, trainer_args, args['TRAIN_DATASET'], args['VALIDATION_DATASET'], "", "", args['IS_USE_TEXT'],
                          args['IS_USE_SOURCES'], args['IS_USE_AUTHORS'], args['IS_USE_TWEETS'], args['IS_USE_RETWEETS'], args['IS_TRUNCATED'], args['IS_ENSEMBLE'], args['RANDOM_STATE'], args['IS_HP_SEARCH'])
elif(args['IS_HP_SEARCH'] or args['IS_TRAIN_VAL_TEST_SPLIT']):
    test_results = trainer.predict(test_dataset)
    print(test_results.metrics)
    results = test_results.metrics
    del results['eval_runtime']
    del results['eval_samples_per_second']
    del results['eval_steps_per_second']
    write_results_to_file(results, trainer.args, args['TRAIN_DATASET'], args['VALIDATION_DATASET'], args['TEXT_TYPE_TRAIN'], args['TEXT_TYPE_TEST'], args['IS_USE_TEXT'],
                          args['IS_USE_SOURCES'], args['IS_USE_AUTHORS'], args['IS_USE_TWEETS'], args['IS_USE_RETWEETS'], args['IS_TRUNCATED'], args['IS_ENSEMBLE'], args['RANDOM_STATE'], args['IS_HP_SEARCH'])
else:
    write_results_to_file(evaluation_results, trainer.args, args['TRAIN_DATASET'], args['VALIDATION_DATASET'], args['TEXT_TYPE_TRAIN'], args['TEXT_TYPE_TEST'], args['IS_USE_TEXT'],
                          args['IS_USE_SOURCES'], args['IS_USE_AUTHORS'], args['IS_USE_TWEETS'], args['IS_USE_RETWEETS'], args['IS_TRUNCATED'], args['IS_ENSEMBLE'], args['RANDOM_STATE'], args['IS_HP_SEARCH'])