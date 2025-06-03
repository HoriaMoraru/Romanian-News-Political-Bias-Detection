import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # suppress the joblib warning

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── CONFIG ─────────────────────────────────────────────────────────────────────
INPUT_MATRIX             = "dataset/ml/entity_domain_sentiment_features.csv"
OUTPUT_CLUSTERS          = "dataset/ml/entity_domain_sentiment_pca_clusters.csv"
OUTPUT_CLUSTER_SUMMARY   = "dataset/ml/entity_domain_pca_clusters_summary.csv"
OUTPUT_PLOT              = "visualization/plots/entity_domain_pca_clusters_2d_plot.png"
N_COMPONENTS             = 2
N_CLUSTERS               = 3
# ────────────────────────────────────────────────────────────────────────────────


def main():
    # 1) LOAD the original entity×domain sentiment CSV
    df = pd.read_csv(INPUT_MATRIX)
    # Expect columns:
    #   ['entity','domain','pos_count','neg_count','neu_count','total_mentions','bias_score','polarity']
    logging.info(f"Loaded {len(df)} rows (entity × domain) from '{INPUT_MATRIX}'")

    # 2) COMPUTE ENTITY-LEVEL IDF for positives & negatives
    #
    #    idf_pos_entity[e] = log( N_domains / (# of domains where e has pos_count>0) )
    #    idf_neg_entity[e] = log( N_domains / (# of domains where e has neg_count>0) )
    #
    # First, figure out how many unique domains exist in the dataset:
    all_domains = df['domain'].unique()
    N_domains = len(all_domains)

    # For each entity, count in how many domains it appears with pos_count>0 (and neg_count>0).
    # We can group by entity and then count distinct domains where pos_count>0.
    df_pos_presence = (
        df[ df['pos_count'] > 0 ]
        .groupby('entity')['domain']
        .nunique()
        .rename('df_pos')  # document frequency for positive
    )
    df_neg_presence = (
        df[ df['neg_count'] > 0 ]
        .groupby('entity')['domain']
        .nunique()
        .rename('df_neg')  # document frequency for negative
    )

    # Some entities may never appear with pos_count>0 or neg_count>0 in any domain.
    # Fill missing with 0 for those.
    all_entities = df['entity'].unique()
    df_pos_presence = df_pos_presence.reindex(all_entities, fill_value=0)
    df_neg_presence = df_neg_presence.reindex(all_entities, fill_value=0)

    # Now compute idf for each entity. Use a tiny epsilon to avoid div-by-zero.
    eps = 1e-9
    idf_pos_entity = np.log( (N_domains / (df_pos_presence + eps)) )
    idf_neg_entity = np.log( (N_domains / (df_neg_presence + eps)) )

    # 3) COMPUTE “TF–IDF” at the entity×domain level
    #
    #    pos_tfidf(entity,domain) = pos_count * idf_pos_entity[entity]
    #    neg_tfidf(entity,domain) = neg_count * idf_neg_entity[entity]
    #
    df['pos_tfidf_entity'] = df['entity'].map(idf_pos_entity) * df['pos_count']
    df['neg_tfidf_entity'] = df['entity'].map(idf_neg_entity) * df['neg_count']

    logging.info("Computed entity-level pos_tfidf and neg_tfidf.")

    # 4) AGGREGATE UP TO DOMAIN LEVEL
    #    After that, we'll compute sentiment ratios and entropy at the domain level.

    domain_features = (
        df.groupby('domain')
          .agg({
              'pos_count': 'sum',
              'neg_count': 'sum',
              'neu_count': 'sum',
              'bias_score': 'mean',
              'polarity': 'mean',
              'pos_tfidf_entity': 'sum',
              'neg_tfidf_entity': 'sum'
          })
          .reset_index()
    )
    logging.info(f"Aggregated to {len(domain_features)} unique domains")

    # 5) DERIVE sentiment ratios & entropy at the DOMAIN level
    domain_features['total_mentions'] = (
        domain_features['pos_count']
        + domain_features['neg_count']
        + domain_features['neu_count']
    ).replace(0, np.nan)

    # sentiment ratios (proportions)
    domain_features['pos_rate'] = domain_features['pos_count'] / domain_features['total_mentions']
    domain_features['neg_rate'] = domain_features['neg_count'] / domain_features['total_mentions']
    domain_features['neu_rate'] = domain_features['neu_count'] / domain_features['total_mentions']

    # sentiment entropy = – sum( p_i * log(p_i) )
    p_pos = domain_features['pos_rate']
    p_neg = domain_features['neg_rate']
    p_neu = domain_features['neu_rate']
    domain_features['entropy'] = -(
        p_pos * np.log(p_pos + eps)
        + p_neg * np.log(p_neg + eps)
        + p_neu * np.log(p_neu + eps)
    )

    # If total_mentions was zero for a domain, fill rates & entropy with 0
    domain_features[['pos_rate','neg_rate','neu_rate','entropy']] = \
        domain_features[['pos_rate','neg_rate','neu_rate','entropy']].fillna(0)

    # Rename the “sum of entity-level tfidf” columns:
    domain_features = domain_features.rename(columns={
        'pos_tfidf_entity': 'pos_tfidf',
        'neg_tfidf_entity': 'neg_tfidf'
    })

    # 6) PREPARE FINAL FEATURE MATRIX for PCA & Clustering
    features = [
        'pos_rate', 'neg_rate', 'entropy', 'pos_tfidf', 'neg_tfidf', 'polarity', 'total_mentions',
    ]
    X = domain_features[features].fillna(0).values

    # 7) STANDARDIZE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 8) PCA → 2 COMPONENTS
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    domain_features['PC1'] = X_pca[:, 0]
    domain_features['PC2'] = X_pca[:, 1]
    logging.info(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")

    # 9) KMEANS on the PCA coordinates
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    domain_features['Cluster'] = kmeans.fit_predict(X_pca)
    logging.info("Completed KMeans clustering of domains")

    # 10) SAVE out the domain_features with PCA + cluster assignments
    domain_features.to_csv(OUTPUT_CLUSTERS, index=False)
    logging.info(f"Saved domain clusters to '{OUTPUT_CLUSTERS}'")

    # 11) COMPUTE & SAVE cluster‐wise summary (in the new feature space)
    cluster_summary = (
        domain_features
        .groupby("Cluster")[features]
        .mean()
        .round(3)
    )
    cluster_summary.to_csv(OUTPUT_CLUSTER_SUMMARY)
    logging.info(f"Saved cluster summary statistics to '{OUTPUT_CLUSTER_SUMMARY}'")

    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(12, 8))

    scatter = sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="Cluster",
        data=domain_features,
        palette="Set2",
        s=100,
        alpha=0.8
    )
    for idx, row in domain_features.iterrows():
        plt.text(
            x=row["PC1"] + 0.02,
            y=row["PC2"] + 0.02,
            s=row["domain"],
            fontdict=dict(color='black', size=8),
            alpha=0.6
        )

    plt.title("Clusters of News Domains Based on Engineered Sentiment Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster", loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    logging.info(f"Saved PCA cluster plot to '{OUTPUT_PLOT}'")
    plt.close()


if __name__ == "__main__":
    main()
