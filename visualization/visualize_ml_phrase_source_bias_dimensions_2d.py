import pandas as pd
import matplotlib.pyplot as plt

DOMAIN_EMBEDDINGS_FILE = "dataset/ml/domain_embeddings.csv"
PLOT_OUTPUT_FILE = "visualization/plots/phrase_source_bias_space_2d.png"

# Load the domain embeddings
domain_embeddings = pd.read_csv(DOMAIN_EMBEDDINGS_FILE, index_col=0)

# We'll use only the first two bias dimensions for 2D visualization
dim1 = "bias_dim_1"
dim2 = "bias_dim_2"

# Create scatter plot
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
