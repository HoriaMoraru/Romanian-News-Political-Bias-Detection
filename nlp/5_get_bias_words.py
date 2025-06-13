import pandas as pd
import numpy as np
import logging
from nlp.ignore_words import ignore_words

from nlp.pipeline.vectorizer import TextVectorizer
from ml.NGramPurger import NGramPurger

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
TOPIC_WORDS_FILE      = "dataset/nlp/topic_words.csv"
DATASET_FILE          = "dataset/romanian_political_articles_v2_nlp.csv"
OUTPUT_DATASET_FILE   = "dataset/romanian_political_articles_v2_nlp_with_topicswords.csv"
OUTPUT_FILE           = "dataset/nlp/phrase_source_bias_words.csv"
NORMALIZED_ENTITIES_MAP = "dataset/ml/normalized_entities.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def words_in_article(row: pd.Series) -> list[str]:
    """Return list of column names where the row value > 0."""
    return list(row.index[row > 0])


if __name__ == "__main__":
    logging.info(f"Reading dataset from {DATASET_FILE}…")
    df = pd.read_csv(DATASET_FILE)

    tv = TextVectorizer()
    X = tv.fit_transform(df["cleantext"])       # shape = (n_docs, n_terms)
    feature_names = np.array(tv.get_feature_names_out())

    topic_words_df = pd.read_csv(TOPIC_WORDS_FILE)
    allowed = set(topic_words_df["word"].unique())

    mask_allowed = np.isin(feature_names, list(allowed))
    filtered_feature_names = feature_names[mask_allowed]

    monograms = [w for w in filtered_feature_names if len(w.split()) == 1]
    bigrams  = [w for w in filtered_feature_names if len(w.split()) == 2]
    trigrams = [w for w in filtered_feature_names if len(w.split()) == 3]
    logging.info(f"Found {len(monograms)} unigrams, {len(bigrams)} bigrams, {len(trigrams)} trigrams")

    df_topics_raw = pd.DataFrame(
        X[:, mask_allowed].toarray(),
        columns=filtered_feature_names,
        index=df.index
    )
    df_topics_raw["source"] = df["source_domain"]
    source_freq_raw = df_topics_raw.groupby("source").sum().drop(columns="source", errors="ignore")

    purger_uni_vs_bi  = NGramPurger(longer_phrases=bigrams, shorter_phrases=monograms, ngram_size=1, threshold=0.5)
    purger_uni_vs_tri = NGramPurger(longer_phrases=trigrams, shorter_phrases=monograms, ngram_size=1, threshold=0.5)
    purger_bi_vs_tri  = NGramPurger(longer_phrases=trigrams, shorter_phrases=bigrams, ngram_size=2, threshold=0.5)

    redundant_unigrams_bigrams = purger_uni_vs_bi.find_redundant(source_freq_raw)
    redundant_unigrams_trigrams = purger_uni_vs_tri.find_redundant(source_freq_raw)
    redundant_bigrams_trigrams = purger_bi_vs_tri.find_redundant(source_freq_raw)
    to_remove = set(redundant_unigrams_bigrams) | set(redundant_unigrams_trigrams) | set(redundant_bigrams_trigrams)
    logging.info(f"Found {len(to_remove)} redundant n-grams to remove.")

    kept = [w for w in filtered_feature_names if w not in to_remove]
    logging.info(f"{len(kept)} phrases remain after purging n-grams.")

    mask_pruned = np.isin(feature_names, kept)
    df_pruned = pd.DataFrame(
        X[:, mask_pruned].toarray(),
        columns=feature_names[mask_pruned],
        index=df.index
    )

    df["bias_words"] = df_pruned.apply(words_in_article, axis=1)

    df.to_csv(OUTPUT_DATASET_FILE, index=False)
    logging.info(f"Saved dataset with per-article bias words to {OUTPUT_DATASET_FILE}")

    df_pruned["source"] = df["source_domain"]
    source_freq_pruned = (
        df_pruned
        .groupby("source")
        .sum()
        .drop(columns="source", errors="ignore")
    )

    # with open(NORMALIZED_ENTITIES_MAP, "r", encoding="utf-8") as f:
    #     entity_map = json.load(f)
    # entity_names = {ent.lower() for ent in entity_map.values()}

    # all_cols = set(source_freq_pruned.columns)
    # to_drop_entities = {c for c in all_cols if c.lower() in entity_names}
    # logging.info(f"Dropping {len(to_drop_entities)} entity-name phrases: {sorted(to_drop_entities)}")

    # cols_to_keep = [c for c in source_freq_pruned.columns if c not in to_drop_entities]

    before = len(source_freq_pruned.columns) + len([c for c in ignore_words if c in source_freq_pruned.columns])
    source_freq_pruned = source_freq_pruned.drop(
        columns=ignore_words,
        errors="ignore"
    )
    after = len(source_freq_pruned.columns)
    logging.info(f"Dropped {before - after} ignored phrases, {after} phrases remain.")

    source_freq_pruned.to_csv(OUTPUT_FILE)
    logging.info(f"Saved pruned source-level phrase frequencies to {OUTPUT_FILE}")
