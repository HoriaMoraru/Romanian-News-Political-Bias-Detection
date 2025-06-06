import re
import torch
import logging
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm
from preprocessing.Preprocessor import Preprocessor
from sentence_transformers import SentenceTransformer

# ───────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────
INPUT_FILE = "dataset/romanian_political_articles_v2_shuffled.csv"
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
TOPIC_INFO = "dataset/nlp/bertopic_info.csv"
OUTPUT_HTML = "visualization/plots/bertopic_topics.html"
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def skip_article(text:str, min_words:int = 30) -> bool:
    """
    Skip articles that are too short or contain only whitespace.
    """
    if not text or not text.strip():
        return True
    words = re.findall(r'\w+', text)
    return len(words) < min_words

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device: {device}")

    logging.info(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    logging.info("Dropping rows with missing text or source...")
    df = df.dropna(subset=["maintext", "source_domain", "url"])

    logging.info("Loading preprocessor...")
    preprocessor = Preprocessor()

    tqdm.pandas(desc="Cleaning...")
    df['cleantext'] = df['maintext'].progress_apply(lambda t: preprocessor.process_nlp(t))

    df = df[~df['cleantext'].apply(skip_article)]
    logging.info(f"Filtered dataset to {len(df)} by removing short articles.")

    logging.info(f"Loading model ({MODEL_NAME})…")
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

    documents = df["cleantext"].astype(str).tolist()

    topic_model = BERTopic(embedding_model=embedding_model)
    topics, probs = topic_model.fit_transform(documents)

    topic_model.get_topic_info().to_csv(TOPIC_INFO)
    fig = topic_model.visualize_topics()
    fig.write_html(OUTPUT_HTML)

