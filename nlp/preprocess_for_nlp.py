import logging
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILE    = "dataset/romanian_political_articles_v2_shuffled.csv"
