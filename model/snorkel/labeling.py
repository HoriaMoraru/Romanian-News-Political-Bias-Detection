import pandas as pd
import json
import logging

from snorkel.labeling import LFAnalysis, labeling_function, PandasLFApplier, ABSTAIN

INPUT_DATASET = "dataset/romanian_political_articles_v2_snorkel.csv"
LABEL_MATRIX   = "dataset/snorkel/label_matrix.csv"

# ───────────────────────────────────────────────────────────────────────────────
# 1. LABEL ENUMERATIONS
# ───────────────────────────────────────────────────────────────────────────────
BIASED    = -1
NEUTRAL   = 0
UNBIASED  = 1

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
    logging.info("Reading dataset...")
    df = pd.read_csv(INPUT_DATASET)

    applier = PandasLFApplier(lfs=labeling_functions)
    L_train = applier.apply(df)

    analysis = LFAnalysis(L=L_train, lfs=labeling_functions).lf_summary()
    logging.info("\n%s", analysis)

    pd.DataFrame(L_train, columns=[lf.name for lf in labeling_functions]).to_csv(
        LABEL_MATRIX, index=False
    )

    print(f"Labeling functions applied; label matrix saved to {LABEL_MATRIX}")
