import logging
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import spacy
from collections import defaultdict
from collections import Counter
from preprocessing.Preprocessor import Preprocessor
from ml.PoissonBiasEmbedder import PoissonBiasEmbedder
from ml.svd_postprocess import svd_postprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILE = "dataset/romanian_political_articles_v2_shuffled.csv"
TOO_GOOD_PHRASES_FILE = "dataset/ml/too_good_phrases_v2.csv"
UBIQUITOUS_PHRASES_FILE = "dataset/ml/ubiquitous_phrases.csv"
MATRIX_FILE = "dataset/ml/phrase_domain_frequency_matrix_v2.csv"


def extract_ngrams_spacy(text, n, nlp):
    doc = nlp(text)
    tokens = [
        token.text for token in doc
        if not token.is_space and not token.is_punct and token.is_alpha
    ]

    return set(' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def preprocess_for_ml(text, nlp):
    """
    Removes all punctuation except sentence-ending periods.
    Replaces each sentence-ending period with the token 'PERIOD'.
    Lowercases the first word of each sentence.
    """

    doc = nlp(text)
    processed_sentences = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # Remove all punctuation from the sentence
        clean_sent = re.sub(r'[^\w\s]', '', sent_text)
        words = clean_sent.split()
        if not words:
            continue

        # Lowercase the first word
        words[0] = words[0].lower()

        processed_sentences.append(" ".join(words))

    return " PERIOD ".join(processed_sentences)

def create_phrase_frequency_matrix(df):
    """
    Creates a phrase-domain frequency matrix (Nij matrix), where:
    - Each row represents a unique phrase (unigram, bigram, trigram, etc.)
    - Each column represents a unique news source domain
    - Each cell [i, j] contains the number of times phrase i appears in articles from domain j

    Returns:
        pd.DataFrame: A phrase-by-domain frequency matrix (rows = phrases, columns = domains)
    """
    all_phrases = list(set(phrase for phrases_list in df['all_ngrams'] for phrase in phrases_list))

    # Create the Nij matrix (rows: phrases, columns: domains)
    domains = df['source_domain'].unique()
    nij_matrix = pd.DataFrame(0, index=all_phrases, columns=domains)

    for domain in domains:
        domain_articles = df[df['source_domain'] == domain]
        domain_phrases = Counter()

        for phrases_list in domain_articles['all_ngrams']:
            domain_phrases.update(phrases_list)

        for phrase, count in domain_phrases.items():
            nij_matrix.at[phrase, domain] = count

    return nij_matrix

def build_reverse_index_by_ngram(longer_phrases, ngram_size):
    """
    Builds a reverse index where keys are subphrases (of size `ngram_size`)
    and values are lists of longer phrases that contain them.

    For example, if `longer_phrases` contains:
        ["partid social democrat", "statul paralel corupt"],
    and `ngram_size = 2`, then one of the index entries will be:
        "partid social" → ["partid social democrat"]

    Returns:
        A dictionary mapping each subphrase to a list of longer phrases that contain it
    """
    reverse_index = defaultdict(list)

    for phrase in longer_phrases:
        tokens = phrase.split()
        for i in range(len(tokens) - ngram_size + 1):
            subphrase = ' '.join(tokens[i:i+ngram_size])
            reverse_index[subphrase].append(phrase)

    return reverse_index

def find_phrases_to_delete_fast(shorter_phrases, reverse_index, nij_matrix, threshold):
    """
    Identifies and returns a list of shorter phrases that should be deleted because they are
    heavily subsumed by longer phrases in terms of frequency.

    This is an optimized version that avoids O(n^2) comparisons by using a reverse index:
    - `reverse_index` is a dictionary mapping each shorter phrase to a list of longer phrases that contain it.
    - For each shorter phrase, the function compares its total count (across all domains) to that of longer phrases.
    - If any longer phrase has a frequency >= `threshold` * shorter phrase frequency, the shorter one is flagged for deletion.

    Parameters:
        shorter_phrases: List or set of shorter phrases (e.g. unigrams or bigrams)
        reverse_index: Maps shorter phrases to the longer phrases that contain them
        nij_matrix: Phrase-domain frequency matrix (rows = phrases, columns = domains)
        threshold: Proportion threshold to consider a shorter phrase redundant

    Returns:
        Phrases to delete (those that are likely redundant due to being contained in more dominant longer phrases)
    """
    phrases_to_delete = []

    for short in shorter_phrases:
        short_count = nij_matrix.loc[short].sum()

        candidates = reverse_index.get(short)

        if not candidates:
            continue

        for long_phrase in candidates:
            long_count = nij_matrix.loc[long_phrase].sum()
            if long_count / short_count >= threshold:
                phrases_to_delete.append(short)
                break

    return phrases_to_delete

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
    monograms = set(mono for monograms_list in df['monograms'] for mono in monograms_list)
    bigrams = set(bi for bigrams_list in df['bigrams'] for bi in bigrams_list)
    trigrams = set(tri for trigrams_list in df['trigrams'] for tri in trigrams_list)

    reverse_index_bigrams = build_reverse_index_by_ngram(bigrams, 1)
    reverse_index_trigrams = build_reverse_index_by_ngram(trigrams, 1)
    reverse_index_bigrams_in_trigrams = build_reverse_index_by_ngram(trigrams, 2)

    monograms_in_bigrams_to_delete = find_phrases_to_delete_fast(monograms, reverse_index_bigrams, phrase_frequency_matrix, threshold)
    monograms_in_trigrams_to_delete = find_phrases_to_delete_fast(monograms, reverse_index_trigrams, phrase_frequency_matrix, threshold)
    bigrams_to_delete = find_phrases_to_delete_fast(bigrams, reverse_index_bigrams_in_trigrams, phrase_frequency_matrix, threshold)

    to_delete = monograms_in_bigrams_to_delete + monograms_in_trigrams_to_delete + bigrams_to_delete

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
    nlp = spacy.load("ro_core_news_sm")

    logging.info("Preprocessing text...")
    tqdm.pandas()
    df['cleantext'] = df['maintext'].progress_apply(
    lambda t: preprocess_for_ml(Preprocessor(t).process(), nlp)
    )

    df = df[df['cleantext'].str.split().str.len() > 30]
    logging.info(f"Filtered dataset size: {len(df)}")

    logging.info("Extracting n-grams...")
    df['monograms'] = df['cleantext'].progress_apply(lambda t: extract_ngrams_spacy(t, 1, nlp))
    df['bigrams'] = df['cleantext'].progress_apply(lambda t: extract_ngrams_spacy(t, 2, nlp))
    df['trigrams'] = df['cleantext'].progress_apply(lambda t: extract_ngrams_spacy(t, 3, nlp))

    logging.info("Removing phrases that contain 'PERIOD'...")
    df['monograms'] = df['monograms'].progress_apply(remove_period_ngrams)
    df['bigrams'] = df['bigrams'].progress_apply(remove_period_ngrams)
    df['trigrams'] = df['trigrams'].progress_apply(remove_period_ngrams)
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
