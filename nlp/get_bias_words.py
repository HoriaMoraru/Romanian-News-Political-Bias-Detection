import pandas as pd
import numpy as np
from nlp.pipeline.vectorizer import TextVectorizer
import logging

TOPIC_WORDS_FILE = "dataset/nlp/topic_words.csv"
DATASET_FILE = "dataset/romanian_political_articles_v2_nlp_with_topics.csv"
OUTPUT_FILE = "dataset/nlp/phrase_source_bias_words.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    df = pd.read_csv(DATASET_FILE)

    tv = TextVectorizer()
    X = tv.fit_transform(df["cleantext"])  # shape = (n_docs, n_terms)
    feature_names = np.array(tv.get_feature_names_out())

    topic_words_df = pd.read_csv(TOPIC_WORDS_FILE)
    allowed = set(topic_words_df["word"].unique())

    mask = np.isin(feature_names, list(allowed))
    filtered_feature_names = feature_names[mask]

    X_topics = X[:, mask]  # still sparse, shape = (n_docs, n_topic_words)
    df_topics = pd.DataFrame(
        X_topics.toarray(),
        columns=filtered_feature_names,
        index=df.index
    )

    source_freq = (
        df_topics
        .assign(source=df["source_domain"])
        .groupby("source")
        .sum()
    )
    source_freq.to_csv(OUTPUT_FILE)
