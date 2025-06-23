import pandas as pd
import logging
import torch
from pprint import pformat

from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling.analysis import LFAnalysis
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.analysis import get_label_buckets

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DATASET = "dataset/romanian_political_articles_v2_snorkel.csv"
LABEL_MATRIX   = "dataset/snorkel/label_matrix.csv"
ARTICLE_LABELS = "dataset/snorkel/article_labels.csv"
GOLD_LABELS   = "manual_labeling/manual_labels.csv"

# ───────────────────────────────────────────────────────────────────────────────
# 1. LABEL ENUMERATIONS
# ───────────────────────────────────────────────────────────────────────────────
ABSTAIN = -1
BIASED   = 0
UNBIASED = 1

# ───────────────────────────────────────────────────────────────────────────────
# 2. LABELING FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

@labeling_function()
def lf_high_bias_word_ratio(x):
    return BIASED if x.bias_word_ratio > 0.09 else ABSTAIN

@labeling_function()
def lf_excessive_exclaims(x):
    return BIASED if x.exclaim_count > 5 else ABSTAIN

@labeling_function()
def lf_excessive_questions(x):
    return BIASED if x.question_count > 5 else ABSTAIN

@labeling_function()
def lf_strong_llama_sentiment_biased(x):
    if x.text_sentiment_llama in ["pozitiv", "negativ"] and x.sentiment_confidence > 0.8:
        return BIASED
    return ABSTAIN

@labeling_function()
def lf_low_topic_entropy(x):
    return BIASED if x.topic_entropy < 3 else ABSTAIN

@labeling_function()
def lf_conditional_hedging(x):
    return BIASED if x.cond_mood_count > 6 else ABSTAIN

@labeling_function()
def lf_topic_sim_to_biased(x):
    return BIASED if x.topic_sim_bias > 0.5 else ABSTAIN

@labeling_function()
def lf_high_entropy_unbiased(x):
    return UNBIASED if x.topic_entropy > 5.0 else ABSTAIN

@labeling_function()
def lf_neutral_llama_sentiment_unbiased(x):
    if x.text_sentiment_llama == "neutru" and x.sentiment_confidence > 0.7:
        return UNBIASED
    return ABSTAIN

@labeling_function()
def lf_clean_unbiased(x):
    if (
        x.bias_word_ratio <= 0.1
        and x.exclaim_count   == 0
        and x.question_count  == 0
        and x.cond_mood_count <= 2
        and x.text_sentiment_llama == "neutru"
    ):
        return UNBIASED
    return ABSTAIN

@labeling_function()
def lf_excess_positive_stance_biased(x):
    return BIASED if x.positive_stance_polarity > 0.3 else ABSTAIN

@labeling_function()
def lf_excess_negative_stance_biased(x):
    return BIASED if x.negative_stance_polarity >= 0.6 else ABSTAIN

@labeling_function()
def lf_balanced_stance_unbiased(x):
    if x.positive_stance_polarity <= 0.3 and x.negative_stance_polarity < 0.6:
        return UNBIASED
    return ABSTAIN

@labeling_function()
def lf_high_first_pronouns(x):
    return BIASED if x.first_pronouns > 4 else ABSTAIN

@labeling_function()
def lf_high_second_pronouns(x):
    return BIASED if x.second_pronouns > 2 else ABSTAIN

@labeling_function()
def lf_short_sentences_unbiased(x):
    return UNBIASED if 5 < x.avg_sentence_length < 12 else ABSTAIN

labeling_functions = [
    lf_high_bias_word_ratio,
    lf_excessive_exclaims,
    lf_excessive_questions,
    lf_strong_llama_sentiment_biased,
    lf_low_topic_entropy,
    lf_conditional_hedging,
    lf_topic_sim_to_biased,
    lf_high_entropy_unbiased,
    lf_neutral_llama_sentiment_unbiased,
    lf_clean_unbiased,
    lf_high_first_pronouns,
    lf_high_second_pronouns,
    lf_short_sentences_unbiased,
    lf_balanced_stance_unbiased,
    lf_excess_positive_stance_biased,
    lf_excess_negative_stance_biased
]

if __name__ == "__main__":
    logging.info("Reading dataset...")
    df = pd.read_csv(INPUT_DATASET)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    applier = PandasLFApplier(lfs=labeling_functions)
    L_train = applier.apply(df)

    analysis = LFAnalysis(L=L_train, lfs=labeling_functions).lf_summary()
    logging.info("\n%s", analysis)

    label_matrix_df = pd.DataFrame(L_train, columns=[lf.name for lf in labeling_functions])
    label_matrix_df["url"] = df["url"]
    label_matrix_df.to_csv(LABEL_MATRIX, index=False)
    logging.info(f"Saved {LABEL_MATRIX}")

    logging.info(f"Labeling functions applied; label matrix saved to {LABEL_MATRIX}")

    label_model = LabelModel(cardinality=2, device="cpu", verbose=True)

    label_model.fit(
        L_train=L_train,
        n_epochs=300,
        lr=0.01,
        log_freq=50,
        seed = 123,
        optimizer="adam"
    )

    preds = label_model.predict(L=L_train, tie_break_policy="random")

    label_mapping = {
        -1: "abstain",
        0 : "biased",
        1 : "unbiased"
    }

    lb = get_label_buckets(preds)
    for (label_val,), indices in lb.items():
        label_str = label_mapping.get(label_val, f"unknown({label_val})")
        logging.info(f"Label: {label_str} → {len(indices)} articles")

    df["snorkel_label"] = preds
    df["snorkel_label"] = df["snorkel_label"].map(label_mapping)

    snorkel_labels = df[["url", "cleantext", "snorkel_label"]]
    snorkel_labels.to_csv(ARTICLE_LABELS, index=False)
    logging.info(f"Saved final article→label file to {ARTICLE_LABELS}")

    logging.info("Computing evaluation metrics using gold labels...:")

    gold_df = pd.read_csv(GOLD_LABELS)

    merged = pd.merge(gold_df, snorkel_labels, on="url", how="inner")

    logging.info(f"Merged dataset size: {len(merged)}")
    out = merged[["url", "label", "snorkel_label"]]

    out.to_csv("dataset/snorkel/merged_labels.csv", index=False)

    le = LabelEncoder()
    y_true = le.fit_transform(merged["label"])
    y_pred = le.fit_transform(merged["snorkel_label"])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }

    logging.info("Final evaluation metrics:\n" + pformat(metrics))

    report = classification_report(y_true, y_pred, target_names=le.classes_)
    logging.info(f"\n{report}")
