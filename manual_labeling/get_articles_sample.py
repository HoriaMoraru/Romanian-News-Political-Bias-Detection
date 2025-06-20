import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from snorkel.labeling import LFAnalysis
from model.snorkel.labeling import labeling_functions
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DATASET = "dataset/romanian_political_articles_v2_snorkel.csv"
OUTPUT_FILE = "dataset/manual_labels_sample.csv"

def main():
    logging.info("Loading dataset and label matrix...")
    df = pd.read_csv(INPUT_DATASET)

    df_by_source = df.groupby("source_domain", group_keys=False).apply(
        lambda x: x.sample(n=min(4, len(x)), random_state=42)
    ).reset_index(drop=True)

    features = df[["bias_word_ratio", "topic_entropy"]].copy()
    features = features.fillna(0)

    bins = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
    binned = bins.fit_transform(features)

    df["bias_bin"] = binned[:, 0]
    df["sentiment_bin"] = binned[:, 1]

    df_diverse = df.groupby(["bias_bin", "sentiment_bin"]).apply(lambda x: x.sample(n=2, random_state=42)).reset_index(drop=True)

    combined = pd.concat([df_by_source, df_diverse], ignore_index=True)
    final_sample = combined.drop_duplicates(subset=["url"], keep="first").sample(n=100, random_state=42)

    final_sample.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Saved {len(final_sample)}-sample set to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
