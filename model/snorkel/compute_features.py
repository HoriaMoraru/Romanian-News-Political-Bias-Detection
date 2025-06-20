import logging
import numpy as np
import pandas as pd
import spacy
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from manual_labeling.known_sources_bias import known_bias

INPUT_DATASET = "dataset/romanian_political_articles_v2_nlp_with_topicswords.csv"
OUTPUT_DATASET = "dataset/romanian_political_articles_v2_snorkel.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ──────────────────────────────────────────────────────────────────────────────
# TEXTUAL FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def feature_text_length(df: pd.DataFrame) -> pd.Series:
    lengths = df['cleantext'].fillna("").str.split().apply(len)
    logging.info("Computed text length feature")
    return lengths


def feature_avg_sentence_length(df: pd.DataFrame) -> pd.Series:
    def avg_sent(text):
        sents = [s for s in text.split('.') if s.strip()]
        if not sents:
            return 0
        return np.mean([len(s.split()) for s in sents])

    avg_lens = df['cleantext'].fillna("").apply(avg_sent)
    logging.info("Computed average sentence length feature")
    return avg_lens
# ──────────────────────────────────────────────────────────────────────────────
# ENTITY FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def feature_entity_count(df: pd.DataFrame) -> pd.Series:
    def count_entities(ner_str):
        try:
            ents = eval(ner_str)
            return len(ents)
        except:
            return 0

    counts = df['ner'].fillna('[]').apply(count_entities)
    logging.info("Computed entity count feature")
    return counts
# ──────────────────────────────────────────────────────────────────────────────
# SENTIMENT FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def feature_stance_polarity(df: pd.DataFrame) -> pd.Series:
    """Aggregate stance polarities: (positive - negative) / total entities."""
    def stance_score(stance_str):
        try:
            st_list = eval(stance_str)
        except:
            return 0
        pos = sum(1 for e in st_list if e['stance'].upper() == 'POZITIV')
        neg = sum(1 for e in st_list if e['stance'].upper() == 'NEGATIV')
        neu = sum(1 for e in st_list if e['stance'].upper() == 'NEUTRU')
        total = pos + neg + neu
        if total == 0:
            return 0
        return abs(pos - neg) / total

    scores = df['stance'].fillna('[]').apply(stance_score)
    logging.info("Computed stance score feature")
    return scores


def feature_negative_polarity(df: pd.DataFrame) -> pd.Series:
    """Aggregate stance polarities: (positive - negative) / total entities."""
    def stance_score(stance_str):
        try:
            st_list = eval(stance_str)
        except:
            return 0
        pos = sum(1 for e in st_list if e['stance'].upper() == 'POZITIV')
        neg = sum(1 for e in st_list if e['stance'].upper() == 'NEGATIV')
        neu = sum(1 for e in st_list if e['stance'].upper() == 'NEUTRU')
        total = pos + neg + neu
        if total == 0:
            return 0
        return neg / total

    scores = df['stance'].fillna('[]').apply(stance_score)
    logging.info("Computed polarity feature")
    return scores


def feature_positive_polarity(df: pd.DataFrame) -> pd.Series:
    """Aggregate stance polarities: (positive - negative) / total entities."""
    def stance_score(stance_str):
        try:
            st_list = eval(stance_str)
        except:
            return 0
        pos = sum(1 for e in st_list if e['stance'].upper() == 'POZITIV')
        neg = sum(1 for e in st_list if e['stance'].upper() == 'NEGATIV')
        neu = sum(1 for e in st_list if e['stance'].upper() == 'NEUTRU')
        total = pos + neg + neu
        if total == 0:
            return 0
        return pos / total

    scores = df['stance'].fillna('[]').apply(stance_score)
    logging.info("Computed positive polarity feature")
    return scores


def feature_doc_sentiment(df, sentiment_pipe, nlp):
    def doc_score(text):
        if not text.strip():
            return 0.0

        doc = nlp(text)
        scores = []
        for sent in doc.sents:
            sent_txt = sent.text.strip()
            if not sent_txt:
                continue
            res = sentiment_pipe(
                sent_txt,
                truncation=True,
                max_length=512,
            )[0]
            sign = 1 if res["label"].startswith("POS") else -1
            scores.append(sign * res["score"])

        return float(np.mean(scores)) if scores else 0.0

    doc_sentiment = df["cleantext"].fillna("").apply(doc_score)
    logging.info("Computed doc sentiment feeature.")
    return doc_sentiment
# ──────────────────────────────────────────────────────────────────────────────
# LEXICAL FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def feature_conditional_mood_count(df, nlp):
    """Count tokens in Conditional or Subjunctive mood as a proxy for hedging."""
    def count_conditional(text):
        doc = nlp(text)
        return sum(1 for t in doc
                   if t.pos_ == "AUX" and ("Cnd" in t.morph.get("Mood") or "Sub" in t.morph.get("Mood")))
    cc = df["cleantext"].fillna("").apply(count_conditional)
    logging.info("Completed conditial mood count feature.")
    return cc

def feature_question_exclam_count(df):
    """Count of question marks and exclamation points outside quotes."""
    import re
    def count_punct(text):
        no_quotes = re.sub(r'["\'„”‘’«»]{1,10}[^"\'„”‘’«»]{1,1000}?["\'„”‘’«»]{1,10}', "", text)
        return no_quotes.count("?"), no_quotes.count("!")
    counts = df["maintext"].fillna("").apply(count_punct)
    q_counts = counts.apply(lambda x: x[0])
    e_counts = counts.apply(lambda x: x[1])
    df["question_count"] = q_counts
    df["exclaim_count"] = e_counts
    qe = df[["question_count", "exclaim_count"]]
    logging.info("Computed question-exclamation count feature.")
    return qe


def feature_first_person_pronouns(df: pd.DataFrame, nlp) -> pd.Series:
    """Count occurrences of first-person pronouns (using spaCy morphological features)."""
    def count_first(text):
        doc = nlp(text)
        return sum(1 for token in doc
                   if token.pos_ == "PRON" and
                   token.morph.get("Person") and
                   "1" in token.morph.get("Person"))

    counts = df['cleantext'].fillna("").apply(count_first)
    logging.info("Computed first-person pronoun count feature")
    return counts


def feature_second_person_pronouns(df: pd.DataFrame, nlp) -> pd.Series:
    """Count occurrences of second-person pronouns (using spaCy morphological features)."""
    def count_second(text):
        doc = nlp(text)
        return sum(1 for token in doc
                   if token.pos_ == "PRON" and
                   token.morph.get("Person") and
                   "2" in token.morph.get("Person"))

    counts = df['cleantext'].fillna("").apply(count_second)
    logging.info("Computed second-person pronoun count feature")
    return counts

# ──────────────────────────────────────────────────────────────────────────────
# KNOWN BIAS FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def feature_bias_word_count(df: pd.DataFrame) -> pd.Series:
    def count_bias(bias_list_str, text):
        try:
            bias_list = eval(bias_list_str)
        except:
            return 0
        text_lower = text.lower()
        return sum(text_lower.count(w.lower()) for w in bias_list)

    counts = df.apply(lambda row: count_bias(row['bias_words'], row['cleantext']), axis=1)
    logging.info("Computed bias word count feature")
    return counts


def feature_bias_word_ratio(df: pd.DataFrame) -> pd.Series:
    """Compute ratio of bias words occurrences to total words."""
    bias_counts = feature_bias_word_count(df)
    total_words = feature_text_length(df)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(total_words > 0, bias_counts / total_words, 0)
    logging.info("Computed bias word ratio feature")
    return pd.Series(ratios)


def feature_source_bias_flag(df):
    """Binary flag if source is in known_source_bias and labeled 'biased'."""
    bias = df["source_domain"].apply(lambda s: 1 if known_bias.get(s) == "biased" else 0)
    logging.info("Computed source bias flag feature")
    return bias
# ──────────────────────────────────────────────────────────────────────────────
# TOPIC FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def feature_topic_similarity_to_biased(df, topic_cols, biased_centroid):
    """Cosine similarity of each article's topic vector to the biased-source centroid."""
    sims = cosine_similarity(df[topic_cols].values, biased_centroid)
    ts = pd.Series(sims.flatten(), index=df.index)
    logging.info("Computed feature topic similarity feature.")
    return ts


def feature_topic_entropy(df, topic_cols):
    """Entropy of the BERTopic distribution."""
    def entropy(probs):
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    te = df[topic_cols].apply(entropy, axis=1)
    logging.info("Computed topic entropy feature.")
    return te
# ──────────────────────────────────────────────────────────────────────────────
# FEATURES END
# ──────────────────────────────────────────────────────────────────────────────
def compute_biased_topic_centroid(df, topic_cols):
    """Compute the average topic vector for all 'biased' sources."""
    biased = df[df["source_domain"].isin(
        [s for s, lbl in known_bias.items() if lbl == "biased"]
    )]
    return biased[topic_cols].mean().values.reshape(1, -1)

def assemble_feature_matrix(df: pd.DataFrame, nlp, sentiment, topic_cols) -> pd.DataFrame:
    df['text_length']           = feature_text_length(df)
    df['avg_sentence_length']   = feature_avg_sentence_length(df)
    df['bias_word_count']       = feature_bias_word_count(df)
    df['bias_word_ratio']       = feature_bias_word_ratio(df)
    df['entity_count']          = feature_entity_count(df)
    df['polarity']              = feature_stance_polarity(df)
    df['positive_polarity']     = feature_positive_polarity(df)
    df['negative_polarity']     = feature_negative_polarity(df)
    df['first_pronouns']        = feature_first_person_pronouns(df, nlp)
    df['second_pronouns']       = feature_second_person_pronouns(df, nlp)
    df['cond_mood_count']       = feature_conditional_mood_count(df, nlp)
    q_e                         = feature_question_exclam_count(df)
    df = pd.concat([df, q_e], axis=1)
    df['overall_sentiment']     = feature_doc_sentiment(df, sentiment, nlp)
    df['topic_entropy']         = feature_topic_entropy(df, topic_cols)
    df['source_known_biased']   = feature_source_bias_flag(df)
    biased_centroid             = compute_biased_topic_centroid(df, topic_cols)
    logging.info("Computed biased centroid.")
    df["topic_sim_bias"]        = feature_topic_similarity_to_biased(df, topic_cols, biased_centroid)

    logging.info("Assembled feature matrix with features")
    return df


def main():
    logging.info("Loading dataset...")
    df = pd.read_csv(INPUT_DATASET, parse_dates=['date_publish'])
    logging.info(f"Loaded {len(df)} articles from {INPUT_DATASET}")

    logging.info("Loading romanian nlp model...")
    nlp = spacy.load("ro_core_news_lg")

    logging.info("Loading sentiment pipeline (romanian)...")
    sentiment = pipeline("sentiment-analysis", model="readerbench/RoBERT-large")

    topic_cols = [c for c in df.columns if c.startswith("topic_")]

    feat_matrix = assemble_feature_matrix(df, nlp, sentiment, topic_cols)

    feat_matrix.to_csv(OUTPUT_DATASET, index=False)
    logging.info(f"Saved feature matrix to {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()
