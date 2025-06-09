import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PHRASE_EMBEDDINGS_FILE = "dataset/ml/phrase_domain_phrase_embeddings.csv"
PHRASE_TFIDF_FILE = "dataset/ml/phrase_tfidf_matrix.csv"  # <-- your TF–IDF matrix

def print_top_phrases(embeddings: pd.DataFrame,
                      dim: str,
                      n: int = 10) -> None:

    if dim not in embeddings.columns:
        raise ValueError(f"Column '{dim}' not found in DataFrame. Available columns: {list(embeddings.columns)}")

    sorted_vals = embeddings[dim].sort_values(ascending=False)
    top = sorted_vals.head(n)

    print(f"\n=== Top {n} phrases by '{dim}' ===")
    print(f"{'Rank':>4}  {'Phrase/Domain':<40}  {dim:>10}")
    print("-" * (4 + 2 + 40 + 2 + len(dim) + 2))
    for rank, (label, value) in enumerate(top.items(), start=1):
        print(f"{rank:>4d}. {label:<40}  {value:>+10.4f}")
    print()


def get_top_n_by_dim(embeddings: pd.DataFrame,
                     dim: str,
                     n: int = 500) -> pd.DataFrame:

    if dim not in embeddings.columns:
        raise ValueError(f"Dimension '{dim}' not found in DataFrame columns {list(embeddings.columns)}")

    top_indices = embeddings[dim].sort_values(ascending=False).head(n).index
    return embeddings.loc[top_indices].copy()


if __name__ == "__main__":
    logging.info(f"Loading phrase embeddings from {PHRASE_EMBEDDINGS_FILE}...")
    phrase_embeddings = pd.read_csv(PHRASE_EMBEDDINGS_FILE, index_col=0)

    logging.info(f"Loading TF-IDF matrix from {PHRASE_TFIDF_FILE}...")
    phrase_tfidf = pd.read_csv(PHRASE_TFIDF_FILE, index_col=0)

    # Compute each phrase's highest TF–IDF across all media sources
    logging.info("Computing max TF-IDF per phrase...")
    max_tfidf = phrase_tfidf.max(axis=1).rename("tfidf")

    phrase_embeddings = phrase_embeddings.join(max_tfidf, how="inner")

    DIMENSION_1 = "bias_dim_1"
    DIMENSION_2 = "bias_dim_2"

    print_top_phrases(phrase_embeddings, dim=DIMENSION_1, n=10)
    print_top_phrases(phrase_embeddings, dim=DIMENSION_2, n=10)

    top_500_dim1 = get_top_n_by_dim(phrase_embeddings, DIMENSION_1, n=500)
    top_500_dim2 = get_top_n_by_dim(phrase_embeddings, DIMENSION_2, n=500)

    top_100_dim1_by_tfidf = get_top_n_by_dim(top_500_dim1, "tfidf", n=100)
    top_100_dim2_by_tfidf = get_top_n_by_dim(top_500_dim2, "tfidf", n=100)

    print_top_phrases(top_100_dim1_by_tfidf, dim="tfidf", n=10)
    print_top_phrases(top_100_dim2_by_tfidf, dim="tfidf", n=10)
