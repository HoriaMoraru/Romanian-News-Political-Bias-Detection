import os
import logging

import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
INPUT_CLUSTERS = "dataset/ml/entity_source_pca_clusters.csv"
OUTPUT_PLOT    = "visualization/plots/entity_source_sentiment_pca_clusters.png"
FIG_SIZE       = (10, 7)
POINT_SIZE     = 80
LABEL_OFFSET   = 0.02
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def plot_clusters(df: pd.DataFrame):
    """Scatter PC1 vs PC2, color by Cluster, with domain labels."""
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.figure(figsize=FIG_SIZE)

    for cluster in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cluster]
        plt.scatter(
            sub["PC1"],
            sub["PC2"],
            label=f"Cluster {cluster}",
            s=POINT_SIZE,
            alpha=0.7
        )

    for _, row in df.iterrows():
        plt.text(
            row["PC1"] + LABEL_OFFSET,
            row["PC2"] + LABEL_OFFSET,
            row["domain"],
            fontsize=8,
            alpha=0.8
        )

    plt.legend(title="Cluster")
    plt.title("Domain Clusters in PCA Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    logging.info(f"Saved cluster plot to {OUTPUT_PLOT}")
    plt.close()


def main():
    logging.info(f"Loading clustered data from {INPUT_CLUSTERS}...")
    df = pd.read_csv(INPUT_CLUSTERS)

    plot_clusters(df)


if __name__ == "__main__":
    main()
