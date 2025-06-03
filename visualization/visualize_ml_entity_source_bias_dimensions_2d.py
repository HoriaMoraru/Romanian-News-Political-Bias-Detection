import pandas as pd
import matplotlib.pyplot as plt

DOMAIN_EMBEDDINGS_FILE = "dataset/ml/domain_entity_embeddings.csv"
PLOT_OUTPUT_FILE = "visualization/plots/entity_media_domains_bias_space.png"

domain_embeddings = pd.read_csv(DOMAIN_EMBEDDINGS_FILE, index_col="domain")

# We'll use only the first two bias dimensions for 2D visualization
dim1 = "bias_dim_0"
dim2 = "bias_dim_1"

plt.figure(figsize=(12, 8))
plt.scatter(domain_embeddings[dim1], domain_embeddings[dim2], alpha=0.8)

for domain, row in domain_embeddings.iterrows():
    plt.text(row[dim1] + 0.005, row[dim2], domain, fontsize=8)

plt.title("Media domains in latent bias space (first 2 dimensions)")
plt.xlabel("Bias Dimension 1")
plt.ylabel("Bias Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_OUTPUT_FILE, dpi=300)
