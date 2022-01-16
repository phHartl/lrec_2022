# %%
import pandas as pd
import glob
import json
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from utils import PATH_TO_DATASET
from nltk.tokenize import word_tokenize
from matplotlib.ticker import AutoMinorLocator

# %%
data = pd.read_csv("results_within_datasets.csv")
data.describe()
# %%
df_politifact = data.iloc[:180]
df_politifact.describe()
# %%
df_gossipcop = data.iloc[180:]
df_gossipcop.describe()
#%% Baseline for politifact
baseline_politifact = df_politifact[:10]
df_politifact = df_politifact[10:]
baseline_politifact.describe()
#%% Baseline for gossipcop
baseline_gossipcop = df_gossipcop[:10]
df_gossipcop = df_gossipcop[10:]
baseline_gossipcop.describe()
# %%
data = df_politifact.append(df_gossipcop)
#%%
data_grouped = data.groupby(
    [
        "train_dataset",
        "test_dataset",
        "text_type_train",
        "text_type_test",
        "is_use_text",
        "is_use_sources",
        "is_use_authors",
        "is_use_tweets",
        "is_use_retweets",
        "is_truncated",
        "is_ensemble",
    ]
)
data_grouped["eval_f1"].describe()
#%%
for key, value in data_grouped:
    print("Config: " + str(key))
    print("Accuracy: " + str(round(value["eval_accuracy"].mean(), 3)))
    print("Precision: " + str(round(value["eval_precision"].mean(), 3)))
    print("Recall: " + str(round(value["eval_recall"].mean(), 3)))
    print("F1: " + str(round(value["eval_f1"].mean(), 3)))

#%% CT FAN 21
data_ct = pd.read_csv("checkthatlab/results.csv")
data_ct.describe()

data_grouped_ct = data_ct.groupby(["text_type", "is_truncated"])
data_grouped_ct["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"].mean()
# %%
text_lengths_politifact = []
text_lengths_politifact_abstractive = []
text_lengths_politifact_extractive = []
for file_path in tqdm(
    glob.glob(PATH_TO_DATASET + "politifact/real/**/news content.json", recursive=True)
):
    with open(file_path) as f:
        data = json.load(f)
    if data["text"] == "":
        continue
    text_lengths_politifact.append(len(word_tokenize(data["text"])))

    path_abstractive = file_path[:-5] + "_summary_abstractive.json"
    with open(path_abstractive) as f:
        data = json.load(f)
        text_lengths_politifact_abstractive.append(len(word_tokenize(data["text"])))

    path_extractive = file_path[:-5] + "_summary_extractive.json"
    with open(path_extractive) as f:
        data = json.load(f)
        text_lengths_politifact_extractive.append(len(word_tokenize(data["text"])))

for file_path in tqdm(
    glob.glob(PATH_TO_DATASET + "politifact/fake/**/news content.json", recursive=True)
):
    with open(file_path) as f:
        data = json.load(f)
    if data["text"] == "":
        continue
    text_lengths_politifact.append(len(word_tokenize(data["text"])))

    path_abstractive = file_path[:-5] + "_summary_abstractive.json"
    with open(path_abstractive) as f:
        data = json.load(f)
        text_lengths_politifact_abstractive.append(len(word_tokenize(data["text"])))

    path_extractive = file_path[:-5] + "_summary_extractive.json"
    with open(path_extractive) as f:
        data = json.load(f)
        text_lengths_politifact_extractive.append(len(word_tokenize(data["text"])))

#%%
df_politifact_plot = pd.DataFrame(
    {
        "Original texts": text_lengths_politifact,
        "Abstractive summaries": text_lengths_politifact_abstractive,
        "Extractive summaries": text_lengths_politifact_extractive,
    }
)

# Remove outliers
df_politifact_plot = df_politifact_plot[
    (numpy.abs(stats.zscore(df_politifact_plot)) < 3).all(axis=1)
]
print(df_politifact_plot.describe())
# %%
text_lengths_gossipcop = []
text_lengths_gossipcop_abstractive = []
text_lengths_gossipcop_extractive = []
for file_path in tqdm(
    glob.glob(PATH_TO_DATASET + "gossipcop/real/**/news content.json", recursive=True)
):
    with open(file_path) as f:
        data = json.load(f)
    if data["text"] == "":
        continue
    text_lengths_gossipcop.append(len(word_tokenize(data["text"])))

    path_abstractive = file_path[:-5] + "_summary_abstractive.json"
    with open(path_abstractive) as f:
        data = json.load(f)
        text_lengths_gossipcop_abstractive.append(len(word_tokenize(data["text"])))

    path_extractive = file_path[:-5] + "_summary_extractive.json"
    with open(path_extractive) as f:
        data = json.load(f)
        text_lengths_gossipcop_extractive.append(len(word_tokenize(data["text"])))

for file_path in tqdm(
    glob.glob(PATH_TO_DATASET + "gossipcop/fake/**/news content.json", recursive=True)
):
    with open(file_path) as f:
        data = json.load(f)
    if data["text"] == "":
        continue
    text_lengths_gossipcop.append(len(word_tokenize(data["text"])))

    path_abstractive = file_path[:-5] + "_summary_abstractive.json"
    with open(path_abstractive) as f:
        data = json.load(f)
        text_lengths_gossipcop_abstractive.append(len(word_tokenize(data["text"])))

    path_extractive = file_path[:-5] + "_summary_extractive.json"
    with open(path_extractive) as f:
        data = json.load(f)
        text_lengths_gossipcop_extractive.append(len(word_tokenize(data["text"])))

#%%
df_gossipcop_plot = pd.DataFrame(
    {
        "Original texts": text_lengths_gossipcop,
        "Abstractive summaries": text_lengths_gossipcop_abstractive,
        "Extractive summaries": text_lengths_gossipcop_extractive,
    }
)

# Remove outliers
df_gossipcop_plot = df_gossipcop_plot[
    (numpy.abs(stats.zscore(df_gossipcop_plot)) < 3).all(axis=1)
]

print(df_gossipcop_plot.describe())
#%% Do it again for CT-FAN 21
df_ct = pd.read_csv("checkthatlab/testing.csv")
df_ct = df_ct.append(pd.read_csv("checkthatlab/training.csv"))
df_ct["text_length"] = df_ct["text"].apply(lambda x: len(word_tokenize(x)))
df_ct["text_extractive_length"] = df_ct["text_extractive"].apply(
    lambda x: len(word_tokenize(x))
)
df_ct["text_abstractive_length"] = df_ct["text_abstractive"].apply(
    lambda x: len(word_tokenize(x))
)

#%%
# Remove outliers
df_ct = df_ct[(numpy.abs(stats.zscore(df_ct.iloc[:, -3:])) < 3).all(axis=1)]

# Remove other columns and rearrange
df_ct_plot = df_ct.iloc[:, -3:]
df_ct_plot = df_ct_plot[
    ["text_length", "text_abstractive_length", "text_extractive_length"]
]

print(df_ct_plot.describe())
#%% Generate histogram of FakeNewsNet & CT-FAN 21 text length distribution
colors = ["#9b004b", "#10899c", "#9c8c10"]


def get_data(row, counter):
    if row == 0:
        return df_politifact_plot.iloc[:, counter]
    elif row == 1:
        return df_gossipcop_plot.iloc[:, counter]
    elif row == 2:
        return df_ct_plot.iloc[:, counter]


fig = plt.figure(constrained_layout=True)
fig.suptitle(" ")

# create 3x1 subfigs
subfigs = fig.subfigures(nrows=3, ncols=1)
for row, subfig in enumerate(subfigs):
    title = ""
    if row == 0:
        title = "Politifact"
    if row == 1:
        title = "Gossipcop"
    if row == 2:
        title = "CT-FAN 21"

    subfig.suptitle(title, fontweight="bold")

    # create 1x3 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    counter = 0
    for col, ax in enumerate(axs):
        ax.set_xlim([0, 2000])
        if row == 0:
            if counter == 0:
                ax.set_title("Original", color=colors[counter], style="italic")
            elif counter == 1:
                ax.set_title("Abstractive", color=colors[counter], style="italic")
            elif counter == 2:
                ax.set_title("Extractive", color=colors[counter], style="italic")

        hist = sns.histplot(
            get_data(row, counter),
            stat="percent",
            ax=ax,
            color=colors[counter],
            kde=True,
        )
        hist.set_ylabel(None)
        hist.set_xlabel(None)
        counter += 1

fig.supylabel("Percentage", fontweight="bold")
fig.supxlabel("Amount of Tokens", fontweight="bold")

fig.set_size_inches(11.69, 8.27)
fig.set_dpi(400)

plt.show()
#%%
print(
    len(df_politifact_plot[df_politifact_plot["Original texts"] > 500])
    / len(df_politifact_plot)
)
print(
    len(df_gossipcop_plot[df_gossipcop_plot["Original texts"] > 500])
    / len(df_gossipcop_plot)
)
print(len(df_ct_plot[df_ct_plot["text_length"] > 500]) / len(df_ct_plot))
# %% Inferential statistics
#%% Test whether a normal distribution exists for each condition
from scipy.stats import shapiro, wilcoxon, friedmanchisquare, mannwhitneyu
import scikit_posthocs as sp

for key, value in data_grouped:
    if shapiro(value.eval_f1.tolist())[1] < 0.05:
        print("Not normally distributed for: ")
        print(key)
# %% RQ1 FakeNewsNet -> CT-FAN 21 is obvious by only looking at the data in the table
f1_scores = []
for key, value in data_grouped:
    for element in value.is_truncated:
        if element:
            f1_scores.append(value.eval_f1.tolist())
            break
# Flatten list
f1_scores = [float(item) for sublist in f1_scores for item in sublist]
print("T-test results:")
t_statistic, p_value = wilcoxon(
    f1_scores,
    baseline_politifact["eval_f1"].append(baseline_gossipcop["eval_f1"]),
    alternative="greater",
)
print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))

df_baseline_ct = pd.read_csv("checkthatlab/results_base.csv")

f1_scores_truncated = []
for key, value in data_grouped_ct:
    if value.is_truncated.isin([True]).any():
        f1_scores_truncated.append(value.eval_f1.tolist())

f1_scores_truncated = [
    float(item) for sublist in f1_scores_truncated for item in sublist
]

t_statistic, p_value = wilcoxon(
    f1_scores_truncated, df_baseline_ct["eval_f1"], alternative="greater"
)
print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))
#%% RQ2 Is there a difference when using a hierarchical approach vs without one? -> no significant difference
# Politifact
f1_scores_truncated = []
for key, value in data_grouped:
    if "politifact" in key:
        if value.is_truncated.isin([True]).any():
            f1_scores_truncated.append(value.eval_f1.tolist())

f1_scores_truncated = [
    float(item) for sublist in f1_scores_truncated for item in sublist
]

f1_scores = []
for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["basic"]).any()
            and value.is_use_text.isin([True]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores.append(value.eval_f1.tolist())

f1_scores = [float(item) for sublist in f1_scores for item in sublist]

print("Wilcoxon test results Politifact:")
t_statistic, p_value = wilcoxon(f1_scores, f1_scores_truncated, alternative="greater")
print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))

# Gossipcop
f1_scores_truncated = []
for key, value in data_grouped:
    if "gossipcop" in key:
        if value.is_truncated.isin([True]).any():
            f1_scores_truncated.append(value.eval_f1.tolist())

f1_scores_truncated = [
    float(item) for sublist in f1_scores_truncated for item in sublist
]

f1_scores = []
for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["basic"]).any()
            and value.is_use_text.isin([True]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores.append(value.eval_f1.tolist())

f1_scores = [float(item) for sublist in f1_scores for item in sublist]

print("Wilcoxon test results Gossipcop:")
t_statistic, p_value = wilcoxon(f1_scores, f1_scores_truncated, alternative="greater")
print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))
# %% CT FAN 21 -> significant difference
f1_scores_truncated = []
for key, value in data_grouped_ct:
    if value.is_truncated.isin([True]).any():
        f1_scores_truncated.append(value.eval_f1.tolist())

f1_scores_truncated = [
    float(item) for sublist in f1_scores_truncated for item in sublist
]

f1_scores = []
for key, value in data_grouped_ct:
    if value.is_truncated.isin([False]).any() and value.text_type.isin([0]).any():
        f1_scores.append(value.eval_f1.tolist())

f1_scores = [float(item) for sublist in f1_scores for item in sublist]

print("Wilcoxon test results CT-FAN 21:")
t_statistic, p_value = wilcoxon(f1_scores, f1_scores_truncated, alternative="greater")
print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))
print("Hierarchical transformer representation has an significant impact on CT-FAN 21")
#%% RQ 3 Is there a difference when using a summarization technique?
# -> probably Friedman test with basic, abstractive and extractive as factors
f1_scores_abstractive = []
f1_scores_extractive = []
f1_scores_original = []
for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["abstractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_abstractive.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["extractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_extractive.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["basic"]).any()
            and value.is_use_text.isin([True]).any()
            and (
                value.is_use_sources.isin([False]).any()
                and value.is_use_authors.isin([False]).any()
                and value.is_use_tweets.isin([False]).any()
                and value.is_use_retweets.isin([False]).any()
            )
        ):
            f1_scores_original.append(value.eval_f1.tolist())

f1_scores_abstractive = [
    float(item) for sublist in f1_scores_abstractive for item in sublist
]

f1_scores_extractive = [
    float(item) for sublist in f1_scores_extractive for item in sublist
]

f1_scores_original = [float(item) for sublist in f1_scores_original for item in sublist]

print(
    friedmanchisquare(
        f1_scores_abstractive,
        f1_scores_extractive,
        f1_scores_original,
    )
)

print("No significant difference in performance on Politifact")
#%%
# Gossipcop
f1_scores_abstractive = []
f1_scores_extractive = []
f1_scores_original = []
for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["abstractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_abstractive.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["extractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_extractive.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["basic"]).any()
            and value.is_use_text.isin([True]).any()
            and (
                value.is_use_sources.isin([False]).any()
                and value.is_use_authors.isin([False]).any()
                and value.is_use_tweets.isin([False]).any()
                and value.is_use_retweets.isin([False]).any()
            )
        ):
            f1_scores_original.append(value.eval_f1.tolist())

f1_scores_abstractive = [
    float(item) for sublist in f1_scores_abstractive for item in sublist
]

f1_scores_extractive = [
    float(item) for sublist in f1_scores_extractive for item in sublist
]

f1_scores_original = [float(item) for sublist in f1_scores_original for item in sublist]

print(
    friedmanchisquare(
        f1_scores_abstractive,
        f1_scores_extractive,
        f1_scores_original
    )
)

df_posthoc = pd.DataFrame(
    {
        "abstractive": f1_scores_abstractive,
        "extractive": f1_scores_extractive,
        "original": f1_scores_original,
    }
)

print(sp.posthoc_nemenyi_friedman(df_posthoc))

print(
    "For GossipCop there is a significant difference between original texts and abstractive summaries"
)
#%% CT-FAN 21
f1_scores_abstractive = []
f1_scores_extractive = []
f1_scores_original = []
for key, value in data_grouped_ct:
    if value.is_truncated.isin([False]).any() and value.text_type.isin([1]).any():
        f1_scores_abstractive.append(value.eval_f1.tolist())

for key, value in data_grouped_ct:
    if value.is_truncated.isin([False]).any() and value.text_type.isin([2]).any():
        f1_scores_extractive.append(value.eval_f1.tolist())

for key, value in data_grouped_ct:
    if value.is_truncated.isin([False]).any() and value.text_type.isin([0]).any():
        f1_scores_original.append(value.eval_f1.tolist())

f1_scores_abstractive = [
    float(item) for sublist in f1_scores_abstractive for item in sublist
]

f1_scores_extractive = [
    float(item) for sublist in f1_scores_extractive for item in sublist
]

f1_scores_original = [float(item) for sublist in f1_scores_original for item in sublist]

print(
    friedmanchisquare(f1_scores_abstractive, f1_scores_extractive, f1_scores_original)
)

df_posthoc = pd.DataFrame(
    {
        "abstractive": f1_scores_abstractive,
        "extractive": f1_scores_extractive,
        "original": f1_scores_original,
    }
)

print(sp.posthoc_nemenyi_friedman(df_posthoc))

print(
    "For CT-FAN 21 extractive summaries are high significantly better in comparision to abstractive summaries"
)
print(
    "For CT-FAN 21 original texts are significantly better in comparision to abstractive summaries"
)
#%% RQ 4: Is there a difference when we use an ensemble? Either four factored friedmann test or wilcoxon test between ensemble and not ensemble
#Politifact
f1_scores_abstractive = []
f1_scores_extractive = []
f1_scores_original = []
f1_scores_ensemble = []
for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["abstractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_abstractive.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["extractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_extractive.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["basic"]).any()
            and value.is_use_text.isin([True]).any()
            and (
                value.is_use_sources.isin([False]).any()
                and value.is_use_authors.isin([False]).any()
                and value.is_use_tweets.isin([False]).any()
                and value.is_use_retweets.isin([False]).any()
            )
        ):
            f1_scores_original.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["ensemble"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_ensemble.append(value.eval_f1.tolist())

f1_scores_abstractive = [
    float(item) for sublist in f1_scores_abstractive for item in sublist
]

f1_scores_extractive = [
    float(item) for sublist in f1_scores_extractive for item in sublist
]

f1_scores_original = [float(item) for sublist in f1_scores_original for item in sublist]

f1_scores_ensemble = [float(item) for sublist in f1_scores_ensemble for item in sublist]

print(
    friedmanchisquare(
        f1_scores_abstractive,
        f1_scores_extractive,
        f1_scores_original,
        f1_scores_ensemble,
    )
)

print("No significant difference in performance on Politifact")

#Gossipcop
f1_scores_abstractive = []
f1_scores_extractive = []
f1_scores_original = []
f1_scores_ensemble = []
for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["abstractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_abstractive.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["extractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_extractive.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["basic"]).any()
            and value.is_use_text.isin([True]).any()
            and (
                value.is_use_sources.isin([False]).any()
                and value.is_use_authors.isin([False]).any()
                and value.is_use_tweets.isin([False]).any()
                and value.is_use_retweets.isin([False]).any()
            )
        ):
            f1_scores_original.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["ensemble"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_ensemble.append(value.eval_f1.tolist())

f1_scores_abstractive = [
    float(item) for sublist in f1_scores_abstractive for item in sublist
]

f1_scores_extractive = [
    float(item) for sublist in f1_scores_extractive for item in sublist
]

f1_scores_original = [float(item) for sublist in f1_scores_original for item in sublist]

f1_scores_ensemble = [float(item) for sublist in f1_scores_ensemble for item in sublist]

print(
    friedmanchisquare(
        f1_scores_abstractive,
        f1_scores_extractive,
        f1_scores_original,
        f1_scores_ensemble,
    )
)

df_posthoc = pd.DataFrame(
    {
        "abstractive": f1_scores_abstractive,
        "extractive": f1_scores_extractive,
        "original": f1_scores_original,
        "ensemble": f1_scores_ensemble
    }
)

print(sp.posthoc_nemenyi_friedman(df_posthoc))

# %% RQ 4 part 2
f1_scores_truncated = []
for key, value in data_grouped:
    if "politifact" in key:
        if value.is_truncated.isin([True]).any():
            f1_scores_truncated.append(value.eval_f1.tolist())

f1_scores_truncated = [
    float(item) for sublist in f1_scores_truncated for item in sublist
]

f1_scores = []
for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["ensemble"]).any()
            and value.is_use_text.isin([True]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores.append(value.eval_f1.tolist())

f1_scores = [float(item) for sublist in f1_scores for item in sublist]

print("Wilcoxon test results Politifact:")
t_statistic, p_value = wilcoxon(f1_scores, f1_scores_truncated, alternative="greater")
print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))

# Gossipcop
f1_scores_truncated = []
for key, value in data_grouped:
    if "gossipcop" in key:
        if value.is_truncated.isin([True]).any():
            f1_scores_truncated.append(value.eval_f1.tolist())

f1_scores_truncated = [
    float(item) for sublist in f1_scores_truncated for item in sublist
]

f1_scores = []
for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["ensemble"]).any()
            and value.is_use_text.isin([True]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores.append(value.eval_f1.tolist())

f1_scores = [float(item) for sublist in f1_scores for item in sublist]

print("Wilcoxon test results Gossipcop:")
t_statistic, p_value = wilcoxon(f1_scores, f1_scores_truncated, alternative="greater")
print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))



#%% RQ 5 Is there a difference in performance when using a metadata? -> yes highly significant for both datasets
# Politifact
f1_scores_no_meta = []
for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_truncated.isin([False]).any()
            # and value.text_type_train.isin(["ensemble"]).any()
            and value.is_use_text.isin([True]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_no_meta.append(value.eval_f1.tolist())

f1_scores_no_meta = [float(item) for sublist in f1_scores_no_meta for item in sublist]

f1_scores_metadata = []
for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_truncated.isin([False]).any()
            # and value.text_type_train.isin(["ensemble"]).any()
            and value.is_use_text.isin([True]).any()
            and value.is_use_sources.isin([True]).any()
            and value.is_use_authors.isin([True]).any()
            and value.is_use_tweets.isin([True]).any()
            and value.is_use_retweets.isin([True]).any()
        ):
            f1_scores_metadata.append(value.eval_f1.tolist())

f1_scores_metadata = [float(item) for sublist in f1_scores_metadata for item in sublist]

print("Wilcoxon results Politifact:")
t_statistic, p_value = wilcoxon(f1_scores_no_meta, f1_scores_metadata)
print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))

print("Check which side:")
t_statistic, p_value = wilcoxon(
    f1_scores_metadata, f1_scores_no_meta, alternative="greater"
)

print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))
print("Model performs better on average with metadata")

# Gossipcop
f1_scores_no_meta = []
for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_truncated.isin([False]).any()
            # and not value.text_type_train.isin(["ensemble"]).any()
            and value.is_use_text.isin([True]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_no_meta.append(value.eval_f1.tolist())

f1_scores_no_meta = [float(item) for sublist in f1_scores_no_meta for item in sublist]

f1_scores_metadata = []
for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_truncated.isin([False]).any()
            # and not value.text_type_train.isin(["ensemble"]).any()
            and value.is_use_text.isin([True]).any()
            and value.is_use_sources.isin([True]).any()
            and value.is_use_authors.isin([True]).any()
            and value.is_use_tweets.isin([True]).any()
            and value.is_use_retweets.isin([True]).any()
        ):
            f1_scores_metadata.append(value.eval_f1.tolist())

f1_scores_metadata = [float(item) for sublist in f1_scores_metadata for item in sublist]

print("Wilcoxon results Gossipcop:")
t_statistic, p_value = wilcoxon(f1_scores_no_meta, f1_scores_metadata)
print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))

print("Check which side:")
t_statistic, p_value = wilcoxon(
    f1_scores_metadata, f1_scores_no_meta, alternative="greater"
)

print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))
print("Model performs better on average with metadata")
# %% RQ 6 Which part of the metadata is most important? -> Friedmann test with four factors
f1_scores_sources = []
f1_scores_authors = []
f1_scores_tweets = []
f1_scores_retweets = []
for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([True]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_sources.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([True]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_authors.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([True]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_tweets.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "politifact" in key:
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([True]).any()
        ):
            f1_scores_retweets.append(value.eval_f1.tolist())

f1_scores_sources = [float(item) for sublist in f1_scores_sources for item in sublist]

f1_scores_authors = [float(item) for sublist in f1_scores_authors for item in sublist]

f1_scores_tweets = [float(item) for sublist in f1_scores_tweets for item in sublist]

f1_scores_retweets = [float(item) for sublist in f1_scores_retweets for item in sublist]

print(
    friedmanchisquare(
        f1_scores_sources, f1_scores_authors, f1_scores_tweets, f1_scores_retweets
    )
)

df_posthoc = pd.DataFrame(
    {
        "sources": f1_scores_sources,
        "authors": f1_scores_authors,
        "tweets": f1_scores_tweets,
        "retweets": f1_scores_retweets,
    }
)

print(sp.posthoc_nemenyi_friedman(df_posthoc))

f1_scores_sources = []
f1_scores_authors = []
f1_scores_tweets = []
f1_scores_retweets = []
for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([True]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_sources.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([True]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_authors.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([True]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_tweets.append(value.eval_f1.tolist())

for key, value in data_grouped:
    if "gossipcop" in key:
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([True]).any()
        ):
            f1_scores_retweets.append(value.eval_f1.tolist())

f1_scores_sources = [float(item) for sublist in f1_scores_sources for item in sublist]

f1_scores_authors = [float(item) for sublist in f1_scores_authors for item in sublist]

f1_scores_tweets = [float(item) for sublist in f1_scores_tweets for item in sublist]

f1_scores_retweets = [float(item) for sublist in f1_scores_retweets for item in sublist]

print(
    friedmanchisquare(
        f1_scores_sources, f1_scores_authors, f1_scores_tweets, f1_scores_retweets
    )
)

df_posthoc = pd.DataFrame(
    {
        "sources": f1_scores_sources,
        "authors": f1_scores_authors,
        "tweets": f1_scores_tweets,
        "retweets": f1_scores_retweets,
    }
)

print(sp.posthoc_nemenyi_friedman(df_posthoc))

# %%

# %% RQ 8 Differences in performance based on dataset -> yes model performs significantly better on Politifact
# Should this maybe be done with the best configuration instead of all runs?
f1_scores_politifact = []
for key, value in data_grouped:
    if "politifact" in key:
        if value.is_truncated.isin([False]).any():
            f1_scores_politifact.append(value.eval_f1.tolist())

f1_scores_politifact = [
    float(item) for sublist in f1_scores_politifact for item in sublist
]

f1_scores_gossipcop = []
for key, value in data_grouped:
    if "gossipcop" in key:
        if value.is_truncated.isin([False]).any():
            f1_scores_gossipcop.append(value.eval_f1.tolist())

f1_scores_gossipcop = [
    float(item) for sublist in f1_scores_gossipcop for item in sublist
]

print("Wilcoxon results difference between datasets:")
t_statistic, p_value = wilcoxon(f1_scores_politifact, f1_scores_gossipcop)
print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))

print("Check which side:")
t_statistic, p_value = wilcoxon(
    f1_scores_politifact, f1_scores_gossipcop, alternative="greater"
)
print("Z-statistic: %2f" % (t_statistic))
print("p-value: %2.30f" % (p_value))
print("Model performs better on average on Politifact then on GossipCop")
# %% RQ 8 Differences when training on one domain and applying on the other -> this should be analysed qualitativly with a table probably
data_between_datasets = pd.read_csv("results_between_datasets.csv")
data_between_datasets_grouped = data_between_datasets.groupby(
    [
        "train_dataset",
        "test_dataset",
        "text_type_train",
        "text_type_test",
        "is_use_text",
        "is_use_sources",
        "is_use_authors",
        "is_use_tweets",
        "is_use_retweets",
        "is_truncated",
        "is_ensemble",
    ]
)
data_between_datasets_grouped["eval_f1"].describe()
# %%
for key, value in data_between_datasets_grouped:
    print("Config: " + str(key))
    print("Accuracy: " + str(round(value["eval_accuracy"].mean(), 3)))
    print("Precision: " + str(round(value["eval_precision"].mean(), 3)))
    print("Recall: " + str(round(value["eval_recall"].mean(), 3)))
    print("F1: " + str(round(value["eval_f1"].mean(), 3)))
# %%
f1_scores_abstractive = []
f1_scores_extractive = []
f1_scores_original = []
for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["politifact"]).any():
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["abstractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_abstractive.append(value.eval_f1.tolist())

for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["politifact"]).any():
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["extractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_extractive.append(value.eval_f1.tolist())

for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["politifact"]).any():
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["basic"]).any()
            and value.is_use_text.isin([True]).any()
            and (
                value.is_use_sources.isin([False]).any()
                and value.is_use_authors.isin([False]).any()
                and value.is_use_tweets.isin([False]).any()
                and value.is_use_retweets.isin([False]).any()
            )
        ):
            f1_scores_original.append(value.eval_f1.tolist())

f1_scores_abstractive = [
    float(item) for sublist in f1_scores_abstractive for item in sublist
]

f1_scores_extractive = [
    float(item) for sublist in f1_scores_extractive for item in sublist
]

f1_scores_original = [float(item) for sublist in f1_scores_original for item in sublist]

f1_scores_sources = []
f1_scores_authors = []
f1_scores_tweets = []
f1_scores_retweets = []
for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["politifact"]).any():
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([True]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_sources.append(value.eval_f1.tolist())

for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["politifact"]).any():
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([True]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_authors.append(value.eval_f1.tolist())

for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["politifact"]).any():
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([True]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_tweets.append(value.eval_f1.tolist())

for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["politifact"]).any():
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([True]).any()
        ):
            f1_scores_retweets.append(value.eval_f1.tolist())

f1_scores_sources = [float(item) for sublist in f1_scores_sources for item in sublist]

f1_scores_authors = [float(item) for sublist in f1_scores_authors for item in sublist]

f1_scores_tweets = [float(item) for sublist in f1_scores_tweets for item in sublist]

f1_scores_retweets = [float(item) for sublist in f1_scores_retweets for item in sublist]

print(
    friedmanchisquare(
        f1_scores_abstractive,
        f1_scores_extractive,
        f1_scores_original,
        f1_scores_sources,
        f1_scores_authors,
        f1_scores_tweets,
        f1_scores_retweets
    )
)

df_posthoc = pd.DataFrame(
    {
        "original": f1_scores_original,
        "abstractive": f1_scores_abstractive,
        "extractive": f1_scores_extractive,
        "sources": f1_scores_sources,
        "authors": f1_scores_authors,
        "tweets": f1_scores_tweets,
        "retweets": f1_scores_retweets,
    }
)

print(sp.posthoc_nemenyi_friedman(df_posthoc))
# %%
f1_scores_abstractive = []
f1_scores_extractive = []
f1_scores_original = []
for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["gossipcop"]).any():
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["abstractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_abstractive.append(value.eval_f1.tolist())

for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["gossipcop"]).any():
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["extractive"]).any()
            and value.is_use_sources.isin([False]).any()
        ):
            f1_scores_extractive.append(value.eval_f1.tolist())

for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["gossipcop"]).any():
        if (
            value.is_truncated.isin([False]).any()
            and value.text_type_train.isin(["basic"]).any()
            and value.is_use_text.isin([True]).any()
            and (
                value.is_use_sources.isin([False]).any()
                and value.is_use_authors.isin([False]).any()
                and value.is_use_tweets.isin([False]).any()
                and value.is_use_retweets.isin([False]).any()
            )
        ):
            f1_scores_original.append(value.eval_f1.tolist())

f1_scores_abstractive = [
    float(item) for sublist in f1_scores_abstractive for item in sublist
]

f1_scores_extractive = [
    float(item) for sublist in f1_scores_extractive for item in sublist
]

f1_scores_original = [float(item) for sublist in f1_scores_original for item in sublist]

f1_scores_sources = []
f1_scores_authors = []
f1_scores_tweets = []
f1_scores_retweets = []
for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["gossipcop"]).any():
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([True]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_sources.append(value.eval_f1.tolist())

for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["gossipcop"]).any():
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([True]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_authors.append(value.eval_f1.tolist())

for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["gossipcop"]).any():
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([True]).any()
            and value.is_use_retweets.isin([False]).any()
        ):
            f1_scores_tweets.append(value.eval_f1.tolist())

for key, value in data_between_datasets_grouped:
    if value.train_dataset.isin(["gossipcop"]).any():
        if (
            value.is_use_text.isin([False]).any()
            and value.is_use_sources.isin([False]).any()
            and value.is_use_authors.isin([False]).any()
            and value.is_use_tweets.isin([False]).any()
            and value.is_use_retweets.isin([True]).any()
        ):
            f1_scores_retweets.append(value.eval_f1.tolist())

f1_scores_sources = [float(item) for sublist in f1_scores_sources for item in sublist]

f1_scores_authors = [float(item) for sublist in f1_scores_authors for item in sublist]

f1_scores_tweets = [float(item) for sublist in f1_scores_tweets for item in sublist]

f1_scores_retweets = [float(item) for sublist in f1_scores_retweets for item in sublist]

print(
    friedmanchisquare(
        f1_scores_abstractive,
        f1_scores_extractive,
        f1_scores_original,
        f1_scores_sources,
        f1_scores_authors,
        f1_scores_tweets,
        f1_scores_retweets
    )
)

df_posthoc = pd.DataFrame(
    {
        "original": f1_scores_original,
        "abstractive": f1_scores_abstractive,
        "extractive": f1_scores_extractive,
        "sources": f1_scores_sources,
        "authors": f1_scores_authors,
        "tweets": f1_scores_tweets,
        "retweets": f1_scores_retweets,
    }
)

print(sp.posthoc_nemenyi_friedman(df_posthoc))
