import ast
import json
import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
INPUT_ARTICLES_CSV      = "dataset/romanian_political_articles_v2_sentiment.csv"
NORMALIZED_ENTITIES_MAP = "dataset/ml/normalized_entities.json"
OUTPUT_FEATURE_MATRIX   = "dataset/ml/entity_source_sentiment_features.csv"

EPSILON = 1e-9
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_normalization_map(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_stance_list(
    stance_json: str, norm_map: Dict[str, str]
) -> List[Dict[str, str]]:
    parsed = ast.literal_eval(stance_json)
    out = []
    for rec in parsed:
        if isinstance(rec, dict) and "entity" in rec and "stance" in rec:
            ent = rec["entity"].strip()
            ent_norm = norm_map.get(ent, ent)
            out.append({"entity": ent_norm, "stance": rec["stance"]})
    return out


def compute_entity_domain_counts(
    df: pd.DataFrame
) -> pd.DataFrame:
    counters = defaultdict(lambda: {"pos": 0, "neg": 0, "neu": 0})
    for _, row in df.iterrows():
        dom = row["source_domain"]
        for rec in row["stance"]:
            ent = rec["entity"]
            st  = rec["stance"].upper()
            key = (ent, dom)
            if st == "POZITIV":
                counters[key]["pos"] += 1
            elif st == "NEGATIV":
                counters[key]["neg"] += 1
            elif st == "NEUTRU":
                counters[key]["neu"] += 1

    records = []
    for (ent, dom), cnt in counters.items():
        pos, neg, neu = cnt["pos"], cnt["neg"], cnt["neu"]
        total = pos + neg + neu
        bias_score = (pos - neg) / total if total else 0.0
        polarity   = abs(pos - neg) / total if total else 0.0
        records.append({
            "entity": ent,
            "domain": dom,
            "pos_count": pos,
            "neg_count": neg,
            "neu_count": neu,
            "total_mentions": total,
            "bias_score": bias_score,
            "polarity": polarity,
        })
    return pd.DataFrame.from_records(records)


def compute_idf(series_counts: pd.Series, n_domains: int) -> pd.Series:
    dfreq = (
        series_counts[series_counts > 0]
        .groupby("entity")
        .nunique()
        .reindex(series_counts.index.get_level_values("entity").unique(), fill_value=0)
    )
    return np.log(n_domains / (dfreq + EPSILON))


def main():
    logging.info("Loading articles with stance annotations...")
    df = pd.read_csv(INPUT_ARTICLES_CSV, encoding="utf-8")
    df = df.dropna(subset=["url"])
    logging.info(f" → {len(df)} articles loaded")

    logging.info("Loading normalization map...")
    norm_map = load_normalization_map(NORMALIZED_ENTITIES_MAP)

    logging.info("Normalizing stance lists...")
    df["stance"] = df["stance"].apply(lambda s: normalize_stance_list(s, norm_map))

    logging.info("Computing entity x domain counts...")
    ed = compute_entity_domain_counts(df)
    logging.info(f" → {len(ed)} rows (entity,domain)")

    n_domains = ed["domain"].nunique()
    ed_pos = ed.set_index(["entity", "domain"])["pos_count"]
    ed_neg = ed.set_index(["entity", "domain"])["neg_count"]

    idf_pos = compute_idf(ed_pos, n_domains)
    idf_neg = compute_idf(ed_neg, n_domains)

    logging.info("Computing entity x domain TF-IDF...")
    ed["pos_tfidf"] = ed["pos_count"] * ed["entity"].map(idf_pos)
    ed["neg_tfidf"] = ed["neg_count"] * ed["entity"].map(idf_neg)

    logging.info("Aggregating up to domain level...")
    df_dom = (
        ed.groupby("domain")
          .agg(
              pos_count   = ("pos_count", "sum"),
              neg_count   = ("neg_count", "sum"),
              neu_count   = ("neu_count", "sum"),
              bias_score  = ("bias_score", "mean"),
              polarity    = ("polarity", "mean"),
              pos_tfidf   = ("pos_tfidf", "sum"),
              neg_tfidf   = ("neg_tfidf", "sum"),
          )
          .reset_index()
    )
    logging.info(f" → {len(df_dom)} domains aggregated")

    total = df_dom["pos_count"] + df_dom["neg_count"] + df_dom["neu_count"]
    df_dom["total_mentions"] = total.replace(0, np.nan)
    df_dom["pos_rate"] = df_dom["pos_count"] / df_dom["total_mentions"]
    df_dom["neg_rate"] = df_dom["neg_count"] / df_dom["total_mentions"]
    df_dom["neu_rate"] = df_dom["neu_count"] / df_dom["total_mentions"]

    p = df_dom[["pos_rate", "neg_rate", "neu_rate"]].clip(lower=EPSILON)
    df_dom["entropy"] = - (p * np.log(p)).sum(axis=1)
    df_dom[["pos_rate", "neg_rate", "neu_rate", "entropy"]] = \
        df_dom[["pos_rate", "neg_rate", "neu_rate", "entropy"]].fillna(0)

    logging.info(f"Saving domain feature matrix to {OUTPUT_FEATURE_MATRIX}...")
    df_dom.to_csv(OUTPUT_FEATURE_MATRIX, index=False, encoding="utf-8")
    logging.info("Done.")


if __name__ == "__main__":
    main()
