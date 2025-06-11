import pandas as pd
import numpy as np
from nlp.pipeline.vectorizer import TextVectorizer
import logging
from tqdm import tqdm
import re

from preprocessing.Preprocessor import Preprocessor

TOPIC_WORDS_FILE = "dataset/nlp/topic_words.csv"
DATASET_FILE = "dataset/romanian_political_articles_v2_shuffled.csv"
OUTPUT_FILE = "dataset/nlp/phrase_source_important_words.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def skip_article(text: str, min_words: int = 30) -> bool:
    if not text or not text.strip():
        return True
    return len(re.findall(r"\w+", text)) < min_words

if __name__ == "__main__":
    df = pd.read_csv(DATASET_FILE)
    df = df.dropna(subset=["maintext", "source_domain", "url"])
    logging.info(f"Original row count: {len(df)}")

    logging.info("Cleaning text via Preprocessor()…")
    preprocessor = Preprocessor()
    tqdm.pandas(desc="Cleaning articles")
    df["cleantext"] = df["maintext"].progress_apply(lambda t: preprocessor.process_nlp(t))
    df = df[~df["cleantext"].apply(skip_article)]
    logging.info(f"After filtering short/empty: {len(df)} rows remain")

    tv = TextVectorizer()
    cv = tv.get_vectorizer()
    X = cv.fit_transform(df["cleantext"])  # shape = (n_docs, n_terms)
    feature_names = np.array(cv.get_feature_names_out())

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
