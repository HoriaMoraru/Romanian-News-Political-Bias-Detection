import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
import numpy as np
import logging
import re
from preprocessing.Preprocessor import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DATSET = "dataset/romanian_political_articles_v2_shuffled.csv"
COMPUTED_FREQUENCY_MATRIX_FILE = "dataset/ml/phrase_domain_frequency_matrix.csv"
OUTPUT_TOPIC_CSV = "dataset/nlp/romanian_articles_with_topics.csv"
OUTPUT_TOPIC_INFO = "dataset/nlp/romanian_topic_info.csv"
EMBEDDING_FILE = "dataset/nlp/bert_article_embeddings.npy"

MIN_TOPIC_SIZE = 20    # minimum number of articles per topic
N_GRAM_RANGE = (1, 1)
TOP_N_WORDS = 8        # show top 8 words per topic

def skip_article(text:str, min_words:int = 30) -> bool:
    """
    Skip articles that are too short or contain only whitespace.
    """
    if not text or not text.strip():
        return True
    words = re.findall(r'\w+', text)
    return len(words) < min_words

if __name__ == "__main__":
    logging.info("Reading dataset...")
    df = pd.read_csv(INPUT_DATSET)

    logging.info("Dropping rows with missing text or source...")
    df = df.dropna(subset=["maintext", "source_domain", "url"])

    logging.info("Loading preprocessor...")
    preprocessor = Preprocessor()

    tqdm.pandas(desc="Cleaning...")
    df['cleantext'] = df['maintext'].progress_apply(lambda t: preprocessor.process_nlp(t))

    df = df[~df['cleantext'].apply(skip_article)]
    logging.info(f"Filtered dataset to {len(df)} by removing short articles.")

    logging.info(f"Reading pre-computed phrase-frequency matrix fro, {COMPUTED_FREQUENCY_MATRIX_FILE}...")
    phrase_frequency_matrix = pd.read_csv(COMPUTED_FREQUENCY_MATRIX_FILE, index_col = 0)

    cleantexts = df['cleantext'].astype(str).tolist()
    phrases = phrase_frequency_matrix.index.tolist()

    logging.info(f"Loading embeddings {EMBEDDING_FILE}...")
    embeddings = np.load(EMBEDDING_FILE)
    logging.info(f"Computed embeddings shape: {embeddings.shape}")

    logging.info("Loading vectorizer model...")
    vectorizer_model = CountVectorizer(
        vocabulary=phrases,
        token_pattern=r"(?u)\b\w+(?: \w+)*\b",
        lowercase=True
    )

    logging.info(f"Loading BERTopic with the embedding model...")
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=None,
        hdbscan_model=None,
        vectorizer_model = vectorizer_model,
        n_gram_range=N_GRAM_RANGE,
        top_n_words=TOP_N_WORDS,
        min_topic_size=MIN_TOPIC_SIZE,
        nr_topics="auto",
        low_memory=True
    )

    logging.info("Running BERTopic model on the cleaned texts...")
    topics, probs = topic_model.fit_transform(cleantexts, embeddings=embeddings)

    df_topic = df.copy()
    df_topic["topic"] = topics
    df_topic["topic_Probability"] = [float(p.max()) for p in probs]
    df_topic.to_csv(OUTPUT_TOPIC_CSV, index=False)
    logging.info(f"Saved article→topic CSV to: {OUTPUT_TOPIC_CSV}")

    topic_info = topic_model.get_topic_info()[['Topic', 'Count', 'Name']]
    topic_info.to_csv(OUTPUT_TOPIC_INFO, index=False)
    logging.info(f"Saved topic info to: {OUTPUT_TOPIC_INFO}")
