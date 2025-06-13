import logging
import pandas as pd
import ast
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DATASET    = "dataset/romanian_political_articles_v2_sentiment.csv"
NORMALIZED_ENTITIES_MAP = "dataset/ml/normalized_entities.json"
OUTPUT_DATASET = "dataset/romanian_political_articles_v2_nlp.csv"

def normalize_entity(entity, normalization_dict: dict):
    return normalization_dict.get(entity.strip(), entity.strip())

def normalize_stance_list(stance_json_str, normalization_dict: dict):
    original = ast.literal_eval(stance_json_str)
    normalized = []
    for d in original:
        if isinstance(d, dict) and "entity" in d and "stance" in d:
            normalized.append({
                "entity": normalize_entity(d["entity"], normalization_dict),
                "stance": d["stance"]
            })
    return normalized

if __name__ == "__main__":

    logging.info("Reading dataset...")
    df = pd.read_csv(INPUT_DATASET, encoding="utf-8")

    logging.info("Reading normalization dictionary...")
    with open(NORMALIZED_ENTITIES_MAP, "r", encoding="utf-8") as f:
        normalization_dict = json.load(f)

    tqdm.pandas(desc="Normalizing entities...")
    df["ner"] = df["ner"].progress_apply(lambda ner: [normalize_entity(e, normalization_dict) for e in ast.literal_eval(ner)])

    tqdm.pandas(desc="Normalizing entities in stance...")
    df["stance"] = df["stance"].progress_apply(lambda s: normalize_stance_list(s, normalization_dict))

    df.to_csv(OUTPUT_DATASET, index=False)
    logging.info(f"Saved cleaned dataset to {OUTPUT_DATASET}.")

