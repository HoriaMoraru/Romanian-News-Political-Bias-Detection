import logging
import numpy as np
import pandas as pd
import spacy

INPUT_DATASET = "dataset/romanian_political_articles_v2_nlp_with_topicswords.csv"
OUTPUT_DATASET = "dataset/romanian_political_articles_v2_snorkel"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def feature_stance_polarity(df: pd.DataFrame) -> pd.Series:
    """Aggregate stance polarities: (positive - negative) / total entities."""
    def stance_score(stance_str):
        try:
            st_list = eval(stance_str)
        except:
            return 0
        pos = sum(1 for e in st_list if e['stance'].upper() == 'POSITIV')
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
        pos = sum(1 for e in st_list if e['stance'].upper() == 'POSITIV')
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
        pos = sum(1 for e in st_list if e['stance'].upper() == 'POSITIV')
        neg = sum(1 for e in st_list if e['stance'].upper() == 'NEGATIV')
        neu = sum(1 for e in st_list if e['stance'].upper() == 'NEUTRU')
        total = pos + neg + neu
        if total == 0:
            return 0
        return pos / total

    scores = df['stance'].fillna('[]').apply(stance_score)
    logging.info("Computed positive polarity feature")
    return scores

def feature_negative_polarity(df: pd.DataFrame) -> pd.Series:
    """Aggregate stance polarities: (positive - negative) / total entities."""
    def stance_score(stance_str):
        try:
            st_list = eval(stance_str)
        except:
            return 0
        pos = sum(1 for e in st_list if e['stance'].upper() == 'POSITIV')
        neg = sum(1 for e in st_list if e['stance'].upper() == 'NEGATIV')
        neu = sum(1 for e in st_list if e['stance'].upper() == 'NEUTRU')
        total = pos + neg + neu
        if total == 0:
            return 0
        return neg / total

    scores = df['stance'].fillna('[]').apply(stance_score)
    logging.info("Computed negative polarity feature")
    return scores

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


def assemble_feature_matrix(df: pd.DataFrame, nlp) -> pd.DataFrame:
    df['text_length'] = feature_text_length(df)
    df['avg_sentence_length'] = feature_avg_sentence_length(df)
    df['bias_word_count'] = feature_bias_word_count(df)
    df['bias_word_ratio'] = feature_bias_word_ratio(df)
    df['entity_count'] = feature_entity_count(df)
    df['polarity'] = feature_stance_polarity(df)
    df['positive_polarity'] = feature_positive_polarity(df)
    df['negative_polarity'] = feature_negative_polarity(df)

    logging.info("Assembled feature matrix with features")
    return df


def main():
    logging.info("Loading dataset...")
    df = pd.read_csv(INPUT_DATASET, parse_dates=['date_publish'])
    logging.info(f"Loaded {len(df)} articles from {INPUT_DATASET}")

    logging.info("Loading romanian nlp model...")
    nlp = spacy.load("ro_core_news_lg")

    feat_matrix = assemble_feature_matrix(df, nlp)

    feat_matrix.to_csv(OUTPUT_DATASET, index=False)
    logging.info(f"Saved feature matrix to {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()
