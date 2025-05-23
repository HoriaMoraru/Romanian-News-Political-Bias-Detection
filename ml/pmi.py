import logging
import pandas as pd
from tqdm import tqdm
import spacy
from ..preprocessing.Preprocessor import Preprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILE = "../dataset/romanian_political_articles_v2_shuffled.csv"

if __name__ == "__main__":

    logging.info("Reading dataset...")
    df = pd.read_csv(INPUT_FILE)

    logging.info("Droping rows with missing text or source...")
    df = df.dropna(subset=["maintext", "source_domain"])

    logging.info("Loading spacy model...")
    nlp = spacy.load("ro_core_news_sm")

    logging.info("Preprocessing text...")
    tqdm.pandas()
    df['cleantext'] = df['maintext'].progress_apply(lambda text: Preprocessor(text).process())
    df = df[df['cleantext'].str.split().str.len() > 30]
    logging.info(f"Filtered dataset size: {len(df)}")
