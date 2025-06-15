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
ABSTAIN = -1
BIASED   = 0
UNBIASED = 1

# ───────────────────────────────────────────────────────────────────────────────
# 2. LABELING FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

@labeling_function()
def lf_source_known_biased(x):
    """If the article’s source is flagged as biased → BIASED."""
    return BIASED if x.source_known_biased == 1 else ABSTAIN

@labeling_function()
def lf_high_bias_word_ratio(x):
    """Lots of bias words relative to text length → BIASED."""
    return BIASED if x.bias_word_ratio > 0.03 else ABSTAIN

@labeling_function()
def lf_high_topic_similarity(x):
    """Article’s topic mix is very similar to known-biased centroid → BIASED."""
    return BIASED if x.topic_sim_bias > 0.7 else ABSTAIN

@labeling_function()
def lf_low_topic_entropy(x):
    """Very low topic entropy (overly focused on one topic) → BIASED."""
    return BIASED if x.topic_entropy < 1.0 else ABSTAIN

@labeling_function()
def lf_excess_exclamations(x):
    """More than 2 exclamation marks outside quotes → BIASED."""
    return BIASED if x.exclaim_count > 2 else ABSTAIN

@labeling_function()
def lf_excess_questions(x):
    """More than 2 rhetorical questions outside quotes → BIASED."""
    return BIASED if x.question_count > 2 else ABSTAIN

@labeling_function()
def lf_strong_sentiment(x):
    """Overall document sentiment very strong → BIASED."""
    return BIASED if abs(x.overall_sentiment) > 0.6 else ABSTAIN

@labeling_function()
def lf_first_person_voice(x):
    """Any first-person pronoun in narrative (not quotes) → BIASED."""
    return BIASED if x.first_pronouns > 0 else ABSTAIN

@labeling_function()
def lf_second_person_voice(x):
    """Any second-person pronoun in narrative → BIASED."""
    return BIASED if x.second_pronouns > 0 else ABSTAIN

@labeling_function()
def lf_unbiased_clean(x):
    """
    No bias words, no emphatic punctuation, and near‐neutral sentiment → UNBIASED.
    (a strict “safe” rule)
    """
    if (
        x.bias_word_ratio == 0
        and x.exclaim_count == 0
        and x.question_count == 0
        and abs(x.overall_sentiment) < 0.1
        and x.cond_mood_count == 0
    ):
        return UNBIASED
    return ABSTAIN

labeling_functions = [
    lf_source_known_biased,
    lf_high_bias_word_ratio,
    lf_high_topic_similarity,
    lf_low_topic_entropy,
    lf_excess_exclamations,
    lf_excess_questions,
    lf_strong_sentiment,
    lf_first_person_voice,
    lf_second_person_voice,
    lf_unbiased_clean,
]

if __name__ == "__main__":
    import snorkel
    print("Version:", snorkel.__version__)

    import pkgutil
    print([m.name for m in pkgutil.iter_modules(snorkel.labeling.__path__)])

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
