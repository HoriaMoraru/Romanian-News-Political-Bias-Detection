import pandas as pd
import matplotlib.pyplot as plt

DOMAIN_EMBEDDINGS_FILE = "dataset/ml/phrase_domain_domain_embeddings.csv"
PLOT_OUTPUT_FILE = "visualization/plots/phrase_source_bias_space_2d.png"
DIM1 = "bias_dim_1"
DIM2 = "bias_dim_2"

def main():
    domain_embeddings = pd.read_csv(DOMAIN_EMBEDDINGS_FILE, index_col=0)

    plt.figure(figsize=(12, 8))
    plt.scatter(domain_embeddings[DIM1], domain_embeddings[DIM2], alpha=0.8)

    for domain, row in domain_embeddings.iterrows():
        plt.text(row[DIM1] + 0.005, row[DIM2], domain, fontsize=8)

    plt.title("Media domains in latent bias space (first 2 dimensions)")
    plt.xlabel("Bias Dimension 1")
    plt.ylabel("Bias Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_FILE, dpi=300)

if __name__ == "__main__":
    main()
