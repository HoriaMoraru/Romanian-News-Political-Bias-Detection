import re
import random
import logging

import numpy as np
import pandas as pd
import torch

from bertopic import BERTopic
from tqdm import tqdm
from umap import UMAP
from hdbscan import HDBSCAN

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from preprocessing.Preprocessor import Preprocessor

# ───────────────────────────────────────────────────────────────────────────────
# CONSTANTS & ARGPARSE
# ───────────────────────────────────────────────────────────────────────────────
INPUT_FILE    = "dataset/romanian_political_articles_v2_shuffled.csv"
MODEL_NAME    = "intfloat/multilingual-e5-base"
OUTPUT_MODEL  = "models/bertopic_model"
TOPIC_INFO    = "dataset/nlp/bertopic_info.csv"
OUTPUT_HTML   = "visualization/plots/bertopic_topics.html"
SEED          = 42
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def skip_article(text: str, min_words: int = 30) -> bool:
    if not text or not text.strip():
        return True
    return len(re.findall(r"\w+", text)) < min_words

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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

    embeddings = np.load("dataset/nlp/bert_article_embeddings.npy")
    logging.info(f"Computed embeddings, shape: {embeddings.shape}")

    # umap_model = UMAP(
    #     n_neighbors=15,
    #     n_components=50,
    #     min_dist=0.1,
    #     metric="cosine",
    #     random_state=SEED
    # )
    # logging.info(f"UMAP config: n_neighbors=15, n_components=50, min_dist=0.1, metric='cosine'")

    # hdbscan_model = HDBSCAN(
    #     min_cluster_size=50,
    #     min_samples=20,
    #     metric="euclidean",
    #     cluster_selection_method="eom",
    #     prediction_data=True
    # )
    # logging.info(f"HDBSCAN config: min_cluster_size=50, min_samples=20, metric='euclidean'")

    topic_model = BERTopic(
        embedding_model=None,    # we will pass precomputed embeddings, so embedding_model=None
        umap_model=None,
        hdbscan_model=None,
        calculate_probabilities=True,
        verbose=True
    )

    logging.info(f"Fitting topic model...")
    topics, probs = topic_model.fit_transform(documents, embeddings)

    topic_info = topic_model.get_topic_info()
    print(topic_info)
    n_topics = len(topic_info) - 1  # subtract the “-1” outlier row
    n_outliers = topics.count(-1)
    logging.info(f"Number of topics discovered: {n_topics}")
    logging.info(f"Number of outliers: {n_outliers}")

    topic_info.to_csv(TOPIC_INFO, index=False)
    logging.info(f"Saved topic info to {TOPIC_INFO}")

    all_topics = topic_model.get_topics()
    topic_words_list = []
    for topic_id, word_scores in all_topics.items():
        for word, score in word_scores:
            topic_words_list.append({"topic_id": topic_id, "word": word, "c_tf_idf": score})
    pd.DataFrame(topic_words_list).to_csv("dataset/nlp/topic_words.csv", index=False)
    logging.info("Saved topic words to dataset/nlp/topic_words.csv")

    doc_topics = [{"doc_index": idx, "topic": t, "probability": p}
                  for idx, (t, p) in enumerate(zip(topics, probs))]
    pd.DataFrame(doc_topics).to_csv("dataset/nlp/doc_topic_assignments.csv", index=False)
    logging.info("Saved doc-topic assignments to dataset/nlp/doc_topic_assignments.csv")

    topic_model.save(OUTPUT_MODEL)
    logging.info(f"Saved full model to {OUTPUT_MODEL}")

    top_n = 10
    topic_words = []
    for topic_id in topic_model.get_topic_info().Topic:
        if topic_id == -1:
            continue
        words_scores = topic_model.get_topic(topic_id)  # list of (word, score)
        words = [w for (w, _) in words_scores][:top_n]
        topic_words.append(words)

    tokenized_docs = [doc.split() for doc in documents]

    id2word = Dictionary(tokenized_docs)
    corpus = [id2word.doc2bow(text) for text in tokenized_docs]

    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        dictionary=id2word,
        corpus=corpus,
        coherence="c_v"
    )
    coherence_score = cm.get_coherence()
    logging.info(f"Topic coherence (c_v) = {coherence_score:.4f}")

    if n_topics > 0:
        logging.info("Generating topic visualization…")
        fig = topic_model.visualize_topics()
        fig.write_html(OUTPUT_HTML)
        logging.info(f"Topic visualization saved to {OUTPUT_HTML}")
    else:
        logging.warning("No real topics found (all labeled -1). Skipping visualization.")
