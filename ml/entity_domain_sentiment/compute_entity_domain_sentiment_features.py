import pandas as pd
import json
import ast
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DATASET = "dataset/romanian_political_articles_v2_sentiment.csv"
NORMALIZED_ENTITIES_MAP = "dataset/ml/normalized_entities.json"
OUTPUT_MATRIX = "dataset/ml/entity_domain_sentiment_features.csv"

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

def compute_sentiment_features(df) -> list[dict]:
    counters = defaultdict(lambda: {"pos": 0, "neg": 0, "neu": 0})

    for _, row in df.iterrows():
        domain = row["source_domain"]
        for s in row["stance"]:
            entity = s["entity"]
            stance = s["stance"].upper()
            key = (entity, domain)
            if stance == "POZITIV":
                counters[key]["pos"] += 1
            elif stance == "NEGATIV":
                counters[key]["neg"] += 1
            else:
                counters[key]["neu"] += 1

    # Compute derived metrics
    records = []
    for (entity, domain), counts in counters.items():
        pos, neg, neu = counts["pos"], counts["neg"], counts["neu"]
        total = pos + neg + neu
        bias_score = (pos - neg) / total if total else 0
        polarity = abs(pos - neg) / total if total else 0

        records.append({
            "entity": entity,
            "domain": domain,
            "pos_count": pos,
            "neg_count": neg,
            "neu_count": neu,
            "total_mentions": total,
            "bias_score": bias_score,
            "polarity": polarity
        })

    return records

if __name__ == "__main__":
    logging.info("Reading dataset...")
    df = pd.read_csv(INPUT_DATASET, encoding="utf-8")
    df = df.dropna(subset=["url"])
    logging.info(f"Original row count: {len(df)}")

    logging.info("Reading normalization dictionary...")
    with open(NORMALIZED_ENTITIES_MAP, "r", encoding="utf-8") as f:
        normalization_dict = json.load(f)

    logging.info("Normalizing entities...")
    df["ner"] = df["ner"].apply(lambda ner: [normalize_entity(e, normalization_dict) for e in ast.literal_eval(ner)])

    logging.info("Normalizing entities in stance...")
    df["stance"] = df["stance"].apply(lambda s: normalize_stance_list(s, normalization_dict))

    logging.info("Computing sentiment feature matrix...")
    sentiment_records = compute_sentiment_features(df)
    feature_df = pd.DataFrame(sentiment_records)

    logging.info(f"Saving feature matrix with {len(feature_df)} (entity, domain) pairs.")
    feature_df.to_csv(OUTPUT_MATRIX, index=False, encoding="utf-8")
    print("✅ Feature matrix saved to", OUTPUT_MATRIX)
