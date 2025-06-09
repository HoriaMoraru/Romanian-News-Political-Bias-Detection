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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD

from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation._keybert import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance

from preprocessing.Preprocessor import Preprocessor

import spacy

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATIONS
# ───────────────────────────────────────────────────────────────────────────────
INPUT_FILE    = "dataset/romanian_political_articles_v2_shuffled.csv"
EMBEDDINGS_NPY_FILE = "dataset/nlp/bert_article_embeddings.npy"
MODEL_NAME    = "intfloat/multilingual-e5-base"
TOPIC_INFO    = "dataset/nlp/bertopic_info.csv"
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

def chunk_text(text: str, max_length: int):
    for i in range(0, len(text), max_length):
        yield text[i : i + max_length]

def spacy_tokenizer(doc, nlp, stop_words, lemmatize=False):
    tokens = []
    for piece in chunk_text(doc, nlp.max_length):
        for token in nlp(piece):
            if not token.is_alpha or token.lower_ in stop_words:
                continue
            txt = token.lemma_ if lemmatize else token.text
            tokens.append(txt.lower())
    return tokens

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

    # embeddings = np.load(EMBEDDINGS_NPY_FILE)
    # logging.info(f"Computed embeddings, shape: {embeddings.shape}")

    logging.info("Loading spacy model...")
    nlp = spacy.load("ro_core_news_lg")
    STOP_WORDS = nlp.Defaults.stop_words

    vectorizer_model = CountVectorizer(
        tokenizer=lambda doc: spacy_tokenizer(doc, nlp, STOP_WORDS, lemmatize=False),
        preprocessor=lambda x: x, #Skip
        lowercase=False,
        ngram_range=(1 , 3),
        min_df=3,
        max_df=0.95,
        strip_accents=None
    )

    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True
    )

    # pca_model = PCA(
    #     n_components=100,
    #     random_state=SEED
    # )

    svd_model = TruncatedSVD(
        n_components=50,
        random_state=SEED
    )

    umap_model = UMAP(
        n_neighbors=20,
        n_components=5,
        min_dist=0.3,
        metric="cosine",
        random_state=SEED
    )

    dim_reduce = make_pipeline(svd_model, umap_model)

    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.05,
        prediction_data=True,
        core_dist_n_jobs=1
    )

    # representation_model = KeyBERTInspired(
    #     top_n_words=10,
    #     nr_repr_docs=4,
    #     nr_candidate_words=100,
    #     random_state=SEED
    # )

    representation_model = MaximalMarginalRelevance(
        diversity=0.7
    )

    topic_model = BERTopic(
        embedding_model=MODEL_NAME,
        umap_model=dim_reduce,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        calculate_probabilities=True,
        verbose=True
    )

    logging.info(f"Fitting topic model...")
    topics, probs = topic_model.fit_transform(documents)

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
    pd.DataFrame(topic_words_list).to_csv(TOPIC_WORDS, index=False)
    logging.info(f"Saved topic words to {TOPIC_WORDS}")

    top_n = 10
    analyzer = vectorizer_model.build_analyzer()
    tokenized_docs = [analyzer(doc) for doc in documents]
    id2word = Dictionary(tokenized_docs)

    filtered_topics = []
    dropped_info   = []

    for topic_id in topic_model.get_topic_info().Topic:
        if topic_id == -1:
            continue

        words_scores = topic_model.get_topic(topic_id)  # list of (word, score)
        kept = []
        for word, _ in words_scores:
            if word in id2word.token2id:
                kept.append(word)
            if len(kept) >= top_n:
                break

        if kept:
            filtered_topics.append(kept)
        else:
            dropped_info.append((topic_id, [w for w, _ in words_scores]))

    if dropped_info:
        logging.info(f"Dropped {len(dropped_info)} topics because none of their top words were in the dictionary:")
        for tid, original in dropped_info:
            logging.info(f" • topic {tid}: {original}")

    cm = CoherenceModel(
        topics=filtered_topics,
        dictionary=id2word,
        texts=tokenized_docs,
        coherence="c_v",
        processes=1
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
