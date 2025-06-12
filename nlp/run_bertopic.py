import random
import logging

import numpy as np
import pandas as pd
import torch
import json

from bertopic import BERTopic

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from nlp.pipeline.clustering import create_hdbscan
from nlp.pipeline.dimensionality_reduce import create_umap
from nlp.pipeline.tfidf import create_tfidf
from nlp.pipeline.vectorizer import TextVectorizer
from nlp.pipeline.finetune.fine_tunning import create_representation_model_precomputed_embeddings, create_representation_model

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATIONS
# ───────────────────────────────────────────────────────────────────────────────
INPUT_FILE    = "dataset/romanian_political_articles_v2_nlp.csv"
EMBEDDINGS_DOC_NPY         = "dataset/nlp/bert_article_embeddings.npy"
EMBEDDINGS_WORD_NPY        = "dataset/nlp/bert_word_embeddings.npy"
EMBEDDINGS_WORD_JSON       = "dataset/nlp/bert_word_index.json"
MODEL_NAME    = "intfloat/multilingual-e5-base"
TOPIC_WORDS   = "dataset/nlp/topic_words.csv"
DATASET_WITH_TOPICS = "dataset/romanian_political_articles_v2_nlp_with_topics.csv"
OUTPUT_HTML   = "visualization/plots/bertopic_topics.html"
SEED          = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    documents = df["cleantext"].astype(str).tolist()

    doc_embeddings   = np.load(EMBEDDINGS_DOC_NPY)         # (n_docs, D)
    word_embeddings  = np.load(EMBEDDINGS_WORD_NPY)        # (vocab_size, D)
    token_to_idx = json.load(open(EMBEDDINGS_WORD_JSON, "r"))

    vectorizer = TextVectorizer()

    topic_model = BERTopic(
        embedding_model = None,
        umap_model = create_umap(),
        hdbscan_model = create_hdbscan(),
        vectorizer_model = vectorizer,
        ctfidf_model = create_tfidf(),
        representation_model = create_representation_model(),
        calculate_probabilities=True,
        verbose=True
    )

    logging.info(f"Fitting topic model...")
    topics, probs = topic_model.fit_transform(documents=documents)

    logging.info("Adding topic assignments to dataset...")
    df["topic"] = topics
    df.to_csv(DATASET_WITH_TOPICS, index=False)
    logging.info(f"Saved dataset with topics to {DATASET_WITH_TOPICS}…")

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

    logging.info("Running coherence model...")
    analyzer = vectorizer.build_analyzer()
    tokenized_docs = [analyzer(doc) for doc in documents]
    id2word = Dictionary(tokenized_docs)
    topics_for_coherence = [
        [w for w,_ in all_topics[tid][:10]]
        for tid in all_topics
        if tid != -1
    ]

    cm = CoherenceModel(
        topics=topics_for_coherence,
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
