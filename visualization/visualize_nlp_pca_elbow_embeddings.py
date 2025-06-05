import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "dataset/nlp/bert_article_embeddings_pca_summary.csv"
OUTPUT_PLOT = "visualization/plots/bert_embeddings_pca_explained_variance.png"

if __name__ == "__main__":

    pca_df = pd.read_csv(INPUT_FILE)

    plt.figure(figsize=(10, 6))
    plt.plot(pca_df['Component'], pca_df['Cumulative Variance'], marker='o', linewidth=2)
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance vs Number of Components')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
