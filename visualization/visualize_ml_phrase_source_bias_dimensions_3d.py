import pandas as pd
import matplotlib.pyplot as plt

DOMAIN_EMBEDDINGS_FILE = "dataset/ml/phrase_domain_domain_embeddings.csv"
PLOT_OUTPUT_FILE = "visualization/plots/phrase_source_bias_space_3d.png"

def main():
    domain_embeddings = pd.read_csv(DOMAIN_EMBEDDINGS_FILE, index_col=0)

    x = domain_embeddings.iloc[:, 0]
    y = domain_embeddings.iloc[:, 1]
    z = domain_embeddings.iloc[:, 2]
    labels = domain_embeddings.index

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=50)

    for i in range(len(labels)):
        ax.text(x[i], y[i], z[i], labels[i], fontsize=8)

    ax.set_title("Media domains in latent bias space (first 3 dimensions)")
    ax.set_xlabel("Bias Dimension 1")
    ax.set_ylabel("Bias Dimension 2")
    ax.set_zlabel("Bias Dimension 3")

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_FILE, dpi=300)

if __name__ == "__main__":
    main()
