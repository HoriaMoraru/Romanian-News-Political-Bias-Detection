import logging
import ast
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILE = "dataset/romanian_political_articles_v2_ner_normalized.csv"
MATRIX_FILE = "dataset/ml/entity_domain_frequency_matrix.csv"

if __name__ == "__main__":
    logging.info("Reading dataset...")
    df = pd.read_csv(INPUT_FILE)

    logging.info("Computing entity-domain frequency matrix...")
    df["ner"] = df["ner"].apply(ast.literal_eval)

    df_exploded = df.explode("ner")[["url", "source_domain", "ner"]].dropna()
    df_exploded = df_exploded.rename(columns={"ner": "entity"})

    df_exploded = df_exploded.drop_duplicates(subset=["url", "source_domain", "entity"])

    entity_frequency_matrix = pd.crosstab(df_exploded["entity"], df_exploded["source_domain"])
    logging.info(f"The frequency matrix has: {len(entity_frequency_matrix.index)} elements")

    entity_frequency_matrix.to_csv(MATRIX_FILE, index=True, index_label="entity")
    logging.info(f"Entity-domain frequency matrix saved to {MATRIX_FILE}")
