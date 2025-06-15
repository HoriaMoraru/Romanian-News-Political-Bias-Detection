import pandas as pd
import logging
import torch

from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling.analysis import LFAnalysis
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DATASET = "dataset/romanian_political_articles_v2_snorkel.csv"
LABEL_MATRIX   = "dataset/snorkel/label_matrix.csv"
ARTICLE_LABELS = "dataset/snorkel/article_labels.csv"

# ───────────────────────────────────────────────────────────────────────────────
# 1. LABEL ENUMERATIONS
# ───────────────────────────────────────────────────────────────────────────────
ABSTAIN = -0
BIASED   = 1
UNBIASED = 2

# ───────────────────────────────────────────────────────────────────────────────
# 2. LABELING FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

@labeling_function()
def lf_high_bias_word_ratio(x):
    return BIASED if x.bias_word_ratio > 0.07 else ABSTAIN

@labeling_function()
def lf_excessive_exclaims(x):
    return BIASED if x.exclaim_count > 1 else ABSTAIN

@labeling_function()
def lf_excessive_questions(x):
    return BIASED if x.question_count > 1 else ABSTAIN

@labeling_function()
def lf_strong_sentiment(x):
    return BIASED if abs(x.overall_sentiment) > 0.65 else ABSTAIN

@labeling_function()
def lf_low_topic_entropy(x):
    return BIASED if x.topic_entropy < 1.5 else ABSTAIN

@labeling_function()
def lf_conditional_hedging(x):
    return BIASED if x.cond_mood_count > 2 else ABSTAIN

@labeling_function()
def lf_passive_overuse(x):
    return BIASED if x.passive_ratio > 0.5 else ABSTAIN

@labeling_function()
def lf_topic_sim_to_biased(x):
    return BIASED if x.topic_sim_bias > 0.60 else ABSTAIN

@labeling_function()
def lf_high_entropy_unbiased(x):
    return UNBIASED if x.topic_entropy > 3.0 else ABSTAIN

@labeling_function()
def lf_near_zero_sentiment_unbiased(x):
    return UNBIASED if abs(x.overall_sentiment) < 0.05 else ABSTAIN

@labeling_function()
def lf_clean_unbiased(x):
    if (
        x.bias_word_ratio == 0
        and x.exclaim_count   == 0
        and x.question_count  == 0
        and x.cond_mood_count <= 1
        and abs(x.overall_sentiment) < 0.10
    ):
        return UNBIASED
    return ABSTAIN

labeling_functions = [
    lf_high_bias_word_ratio,
    lf_excessive_exclaims,
    lf_excessive_questions,
    lf_strong_sentiment,
    lf_low_topic_entropy,
    lf_conditional_hedging,
    lf_passive_overuse,
    lf_topic_sim_to_biased,
    lf_high_entropy_unbiased,
    lf_near_zero_sentiment_unbiased,
    lf_clean_unbiased,
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

    label_model = LabelModel(cardinality=2, device="gpu", verbose=True)

    label_model.fit(
        L_train=L_train,
        device=device,
        n_epochs=300,
        lr=0.01,
        log_freq=50,
        seed = 123,
        optimizer="adam"
    )

    preds = label_model.predict(L=L_train)

    df["snorkel_label"] = preds

    out = df[["url", "snorkel_label"]]
    out.to_csv(ARTICLE_LABELS, index=False)
    logging.info(f"Saved final article→label file to {ARTICLE_LABELS}")
