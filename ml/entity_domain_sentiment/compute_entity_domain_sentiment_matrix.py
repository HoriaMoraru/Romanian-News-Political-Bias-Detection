import pandas as pd
import json
import ast
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DATASET = "dataset/romanian_political_articles_v2_sentiment.csv"
NORMALIZED_ENTITIES_MAP = "dataset/ml/normalized_entities.json"
OUTPUT_MATRIX = "dataset/ml/entity_domain_sentiment_matrix.csv"

def normalize_entity(entity, normalization_dict: dict):
        return normalization_dict.get(entity.strip(), entity.strip())

def normalize_stance_list(stance_json_str, normalization_dict: dict):
    original = ast.literal_eval(stance_json_str)

    normalized = []

    # In the sentiment dataset, some entities were labeled as "NO_ENTITY", mistakenly, withing a tuple, need to clean that.
    for d in original:
        if isinstance(d, dict) and "entity" in d and "stance" in d:
            normalized.append({
                "entity": normalize_entity(d["entity"], normalization_dict),
                "stance": d["stance"]
            })

    return normalized

def compute_entity_domain_stance_scores(df, stance_calculator: dict) -> dict:
    scores = defaultdict(list)

    for _, row in df.iterrows():
        domain = row["source_domain"]
        for s in row["stance"]:
            entity = s["entity"]
            stance = stance_calculator.get(s["stance"].upper(), 0)
            scores[(entity, domain)].append(stance)

    return scores

if __name__ == "__main__":
    logging.info("Reading dataset...")
    df = pd.read_csv(INPUT_DATASET, encoding="utf-8")

    logging.info("Reading normalization dictionary...")
    with open(NORMALIZED_ENTITIES_MAP, "r", encoding="utf-8") as f:
        normalization_dict = json.load(f)

    logging.info("Normalizing entities...")
    df["ner"] = df["ner"].apply(lambda ner: [normalize_entity(e, normalization_dict) for e in ast.literal_eval(ner)])

    logging.info("Normalizing entities in stance...")
    df["stance"] = df["stance"].apply(lambda s: normalize_stance_list(s, normalization_dict))

    logging.info("Computing entity domain sentiment matrix...")
    stance_calculator = {"NEUTRU": 0, "POSITIV": 1, "NEGATIV": -1}

    scores = compute_entity_domain_stance_scores(df, stance_calculator)

    records = [
        {"entity": entity, "domain": domain, "score": sum(vals) / len(vals)}
        for (entity, domain), vals in scores.items()
    ]

    sentiment_matrix_df = pd.DataFrame(records)
    sentiment_matrix_pivot = sentiment_matrix_df.pivot(index="entity", columns="domain", values="score").fillna(0)
    logging.info(f"The sentiment matrix has: {len(sentiment_matrix_pivot.index)} entities.")

    sentiment_matrix_pivot.to_csv(OUTPUT_MATRIX, index=True, encoding="utf-8")
    print("Entity domain sentiment matrix computed and saved to", OUTPUT_MATRIX)
