import re
import random
import logging

import numpy as np
import pandas as pd
import torch

from bertopic import BERTopic
from tqdm import tqdm

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from preprocessing.Preprocessor import Preprocessor

from nlp.pipeline.clustering import create_hdbscan
from nlp.pipeline.dimensionality_reduce import create_umap
from nlp.pipeline.tfidf import create_tfidf
from nlp.pipeline.vectorizer import TextVectorizer
from nlp.pipeline.fine_tunning import create_representation_model

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATIONS
# ───────────────────────────────────────────────────────────────────────────────
INPUT_FILE    = "dataset/romanian_political_articles_v2_shuffled.csv"
MODEL_NAME    = "intfloat/multilingual-e5-base"
TOPIC_WORDS   = "dataset/nlp/topic_words.csv"
OUTPUT_HTML   = "visualization/plots/bertopic_topics.html"
SEED          = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def skip_article(text: str, min_words: int = 30) -> bool:
    if not text or not text.strip():
        return True
    return len(re.findall(r"\w+", text)) < min_words

if __name__ == "__main__":
    logging.info(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["maintext", "source_domain", "url"])
    logging.info(f"Original row count: {len(df)}")

    logging.info("Cleaning text via Preprocessor()…")
    preprocessor = Preprocessor()
    tqdm.pandas(desc="Cleaning articles")
    df["cleantext"] = df["maintext"].progress_apply(lambda t: preprocessor.process_nlp(t))
    df = df[~df["cleantext"].apply(skip_article)]
    logging.info(f"After filtering short/empty: {len(df)} rows remain")

    documents = df["cleantext"].astype(str).tolist()

    vectorizer = TextVectorizer()

    topic_model = BERTopic(
        embedding_model = MODEL_NAME,
        umap_model = create_umap(),
        hdbscan_model = create_hdbscan(),
        vectorizer_model = vectorizer,
        ctfidf_model = create_tfidf(),
        representation_model = create_representation_model(),
        calculate_probabilities=True,
        verbose=True
    )

    logging.info(f"Fitting topic model...")
    topics, probs = topic_model.fit_transform(documents)

    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1  # subtract the “-1” outlier row
    n_outliers = topics.count(-1)
    logging.info(f"Number of topics discovered: {n_topics}")
    logging.info(f"Number of outliers: {n_outliers}")

    all_topics = topic_model.get_topics()
    topic_words_list = []
    for topic_id, word_scores in all_topics.items():
        for word, score in word_scores:
            topic_words_list.append({"topic_id": topic_id, "word": word, "c_tf_idf": score})
    pd.DataFrame(topic_words_list).to_csv(TOPIC_WORDS, index=False)
    logging.info(f"Saved topic words to {TOPIC_WORDS}")

    top_n = 10
    analyzer = vectorizer.build_analyzer()
    tokenized_docs = [analyzer(doc) for doc in documents]
    id2word = Dictionary(tokenized_docs)

    cm = CoherenceModel(
        topics=all_topics,
        dictionary=id2word,
        texts=tokenized_docs,
        coherence="c_v",
        processes=1
    )
    coherence_score = cm.get_coherence()
    logging.info(f"Topic coherence (c_v) = {coherence_score:.4f}")

    logging.info("Generating topic visualization…")
    fig = topic_model.visualize_topics()
    fig.write_html(OUTPUT_HTML)
    logging.info(f"Topic visualization saved to {OUTPUT_HTML}")
