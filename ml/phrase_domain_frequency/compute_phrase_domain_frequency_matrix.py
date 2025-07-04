import logging
import pandas as pd
from tqdm import tqdm
import spacy
from ml.NGramPurger import NGramPurger
from collections import Counter
from preprocessing.Preprocessor import Preprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MIN_TOTAL_FREQ = 5
INPUT_FILE = "dataset/romanian_political_articles_v2_shuffled.csv"
TOO_GOOD_PHRASES_FILE = "dataset/ml/too_good_phrases.csv"
UBIQUITOUS_PHRASES_FILE = "dataset/ml/ubiquitous_phrases.csv"
MATRIX_FILE = "dataset/ml/phrase_domain_frequency_matrix.csv"
BOILERPLATE_PATTERNS_FILE = "ml/phrase_domain_frequency/utils/boilerplate_patterns.txt"

def extract_ngrams_spacy(text: str, n: int, nlp, stopwords: set[str]) -> set[str]:
    """
    Extraction of n-grams (length = n) from `text` using spaCy.

    Only alphabetic tokens or hyphenated sequences of letters are kept.
    We skip any n-gram for which every token (lowercased) is in `stopwords`.
    In other words, we keep an n-gram if at least one subtoken is NOT a stopword.
    """
    doc = nlp(text)
    filtered_tokens: list[str] = []
    is_all_stop: list[bool] = []

    for token in doc:
        txt = token.text

        if txt.replace("-", "").isalpha():
            filtered_tokens.append(txt)
            is_all_stop.append(txt.lower() in stopwords)

    L = len(filtered_tokens)
    if L < n:
        return set()

    ngrams = set()
    for i in range(L - n + 1):
        window_flags = is_all_stop[i : i + n]
        if not all(window_flags):
            phrase = " ".join(filtered_tokens[i : i + n])
            ngrams.add(phrase)

    return ngrams

def create_phrase_frequency_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a phrase-domain frequency matrix (Nij matrix), where:
     - Each row is a unique phrase (unigram, bigram, trigram, …), no matter how rare.
     - Each column is a unique news source domain.
     - Each cell [i, j] = number of times phrase i appears in articles from domain j.
    """
    all_phrases = df["all_ngrams"].explode().unique().tolist()

    domains = df["source_domain"].unique()
    nij_matrix = pd.DataFrame(0, index=all_phrases, columns=domains, dtype=int)

    for domain in domains:
        domain_counter = Counter()
        for phrases_list in df.loc[df["source_domain"] == domain, "all_ngrams"]:
            domain_counter.update(phrases_list)
        for phrase, count in domain_counter.items():
            nij_matrix.at[phrase, domain] = count

    return nij_matrix

def remove_period_ngrams(ngrams):
    return {phrase for phrase in ngrams if "PERIOD" not in phrase}

def filter_redundant_ngrams(phrase_frequency_matrix, df, threshold=0.7):
    """
    Removes redundant n-grams from the phrase frequency matrix.

    This function identifies and drops monograms and bigrams that are heavily part of
    more informative longer n-grams (bigrams or trigrams) across all domains.
    Example:
        If "social democrat" appears in a bigram and "partid social democrat" appears in a trigram,
        then "social democrat" will be removed from the bigram list if it is found in a trigram
        with a frequency that is at least `threshold` times the frequency of "social democrat" in the bigram.

    Parameters:
        phrase_frequency_matrix: The phrase-by-domain frequency matrix (Nij).
        df: The original DataFrame containing 'monograms', 'bigrams', and 'trigrams' columns.

    Returns:
        A filtered phrase frequency matrix with redundant monograms and bigrams removed.
    """
    monograms = {
        mono
        for sublist in df["monograms"]
        for mono in sublist
        if mono in phrase_frequency_matrix.index
    }
    bigrams  = {
        bi
        for sublist in df["bigrams"]
        for bi in sublist
        if bi in phrase_frequency_matrix.index
    }
    trigrams = {
        tri
        for sublist in df["trigrams"]
        for tri in sublist
        if tri in phrase_frequency_matrix.index
    }

    purger_uni_vs_bi  = NGramPurger(longer_phrases=bigrams, shorter_phrases=monograms, ngram_size=1, threshold=threshold)
    purger_uni_vs_tri = NGramPurger(longer_phrases=trigrams, shorter_phrases=monograms, ngram_size=1, threshold=threshold)
    purger_bi_vs_tri  = NGramPurger(longer_phrases=trigrams, shorter_phrases=bigrams, ngram_size=2, threshold=threshold)

    redundant_unigrams_bigrams = purger_uni_vs_bi.find_redundant(phrase_frequency_matrix)
    redundant_unigrams_trigrams = purger_uni_vs_tri.find_redundant(phrase_frequency_matrix)
    redundant_bigrams_trigrams = purger_bi_vs_tri.find_redundant(phrase_frequency_matrix)
    to_delete = redundant_unigrams_bigrams + redundant_unigrams_trigrams + redundant_bigrams_trigrams
    logging.info(f"Deleting {len(to_delete)} redundant n-grams...")

    return phrase_frequency_matrix.drop(index=to_delete, errors='ignore')

def filter_too_good_phrases(nij_matrix, export_path, threshold = 0.9):
    """
    Removes phrases that occur more than `threshold` proportionally in a single domain.

    Parameters:
        nij_matrix: phrase-by-domain frequency matrix
        threshold: dominance threshold (e.g., 0.9)
        export_path: CSV path to save removed phrases

    Returns:
        pd.DataFrame: cleaned nij_matrix with dominant phrases removed
    """
    phrase_totals = nij_matrix.sum(axis=1)
    phrase_max_per_domain = nij_matrix.max(axis=1)
    dominance_ratio = phrase_max_per_domain / phrase_totals

    too_good_phrases = dominance_ratio[dominance_ratio > threshold]

    logging.info(f"Found {len(too_good_phrases)} phrases that are too good (threshold: {threshold})")

    exported_df = nij_matrix.loc[too_good_phrases.index].copy()
    exported_df["dominance_ratio"] = dominance_ratio.loc[exported_df.index]
    exported_df.to_csv(export_path, index_label="phrase")

    logging.info(f"Exported too good phrases to {export_path}")

    return nij_matrix.drop(index=too_good_phrases.index, errors='ignore')

def filter_ubiquitous_phrases(nij_matrix, export_path, threshold=0.9):
    """
    Removes phrases that occur in more than `threshold` proportion of domains.

    Parameters:
        nij_matrix: phrase-by-domain frequency matrix
        threshold: ubiquity threshold (e.g., 0.9)

    Returns:
        Cleaned nij_matrix with ubiquitous phrases removed
    """
    domain_count = nij_matrix.shape[1]
    phrase_ubiquity = (nij_matrix > 0).sum(axis=1) / domain_count

    ubiquitous_phrases = phrase_ubiquity[phrase_ubiquity > threshold]

    logging.info(f"Found {len(ubiquitous_phrases)} ubiquitous phrases (threshold: {threshold})")

    exported_df = nij_matrix.loc[ubiquitous_phrases.index].copy()
    exported_df["phrase_ubiquity"] = phrase_ubiquity.loc[ubiquitous_phrases.index]
    exported_df.to_csv(export_path, index_label="phrase")

    logging.info(f"Exported ubiquitous phrases to {export_path}")

    return nij_matrix.drop(index=ubiquitous_phrases.index, errors='ignore')

if __name__ == "__main__":
    logging.info("Reading dataset...")
    df = pd.read_csv(INPUT_FILE)

    logging.info("Dropping rows with missing text or source...")
    df = df.dropna(subset=["maintext", "source_domain"])

    logging.info("Loading spacy model...")
    nlp = spacy.load("ro_core_news_lg")
    STOP_WORDS = nlp.Defaults.stop_words

    logging.info("Loading preprocessor...")
    preprocessor = Preprocessor(nlp=nlp)

    tqdm.pandas(desc="Cleaning...")
    df['cleantext'] = df['maintext'].progress_apply(lambda t: preprocessor.process_ml(t))

    df = df[df['cleantext'].str.split().str.len() > 30]
    logging.info(f"Filtered dataset size: {len(df)}")

    logging.info("Extracting n-grams...")
    df['monograms'] = df['cleantext'].progress_apply(lambda t: extract_ngrams_spacy(t, 1, nlp, STOP_WORDS))
    df['bigrams'] = df['cleantext'].progress_apply(lambda t: extract_ngrams_spacy(t, 2, nlp, STOP_WORDS))
    df['trigrams'] = df['cleantext'].progress_apply(lambda t: extract_ngrams_spacy(t, 3, nlp, STOP_WORDS))

    logging.info("Removing phrases that contain 'PERIOD'...")
    df['monograms'] = df['monograms'].progress_apply(remove_period_ngrams)
    df['bigrams'] = df['bigrams'].progress_apply(remove_period_ngrams)
    df['trigrams'] = df['trigrams'].progress_apply(remove_period_ngrams)

    logging.info("Combining n-grams into a single set of all n-grams...")
    df['all_ngrams'] = df.progress_apply(lambda row: row['monograms'] | row['bigrams'] | row['trigrams'], axis=1)

    logging.info("Creating phrase frequency matrix...")
    phrase_frequency_matrix = create_phrase_frequency_matrix(df)
    logging.info(f"The frequency matrix has: {len(phrase_frequency_matrix.index)} elements")

    logging.info("Removing redundant ngrams that are contained in longer ngrams...")
    phrase_frequency_matrix = filter_redundant_ngrams(phrase_frequency_matrix, df, threshold=0.7)
    logging.info(f"The frequency matrix has: {len(phrase_frequency_matrix.index)} elements")

    logging.info("Filtering too good phrases...")
    phrase_frequency_matrix = filter_too_good_phrases(phrase_frequency_matrix,
                                                      export_path=TOO_GOOD_PHRASES_FILE,
                                                      threshold=0.9)
    logging.info(f"The frequency matrix has: {len(phrase_frequency_matrix.index)} elements")

    logging.info("Filtering ubiquitous phrases...")
    phrase_frequency_matrix = filter_ubiquitous_phrases(phrase_frequency_matrix,
                                                        export_path=UBIQUITOUS_PHRASES_FILE,
                                                        threshold=0.9)
    logging.info(f"The frequency matrix has: {len(phrase_frequency_matrix.index)} elements")

    phrase_frequency_matrix.to_csv(MATRIX_FILE, index=True, index_label="phrase")
    logging.info(f"Phrase frequency matrix saved to {MATRIX_FILE}")
