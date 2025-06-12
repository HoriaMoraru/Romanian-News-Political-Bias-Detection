from hdbscan import HDBSCAN

def create_hdbscan(
    min_cluster_size: int = 5,
    min_samples: int = 1,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    prediction_data: bool = True,
    core_dist_n_jobs: int = 1
) -> HDBSCAN:

    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=prediction_data,
        core_dist_n_jobs=core_dist_n_jobs
    )
