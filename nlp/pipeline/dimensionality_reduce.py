from umap import UMAP

def create_umap(
    n_neighbors: int = 15,
    n_components: int = 10,
    min_dist: float = 0.05,
    metric: str = "cosine",
    random_state: int = 42
) -> UMAP:

    return UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
