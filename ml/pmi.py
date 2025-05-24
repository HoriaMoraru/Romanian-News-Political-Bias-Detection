import logging
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import spacy
from collections import defaultdict
from collections import Counter
from preprocessing.Preprocessor import Preprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILE = "ml/dataset/romanian_political_articles_v2_shuffled.csv"
TOO_GOOD_PHRASES_FILE = "ml/dataset/too_good_phrases.csv"
MI_FILE = "ml/dataset/mutual_information_scores_v2.csv"
PMI_FILE = "ml/dataset/pointwise_mutual_information_scores.csv"
MATRIX_FILE = "ml/dataset/phrase_domain_frequency_matrix.csv"

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

def find_phrases_to_delete_fast(shorter_phrases, reverse_index, nij_matrix, threshold=0.7):
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

def filter_too_good_phrases(nij_matrix, export_path, threshold = 0.9):
    """
    Removes phrases that occur more than `threshold` proportionally in a single domain.

    Parameters:
        nij_matrix (pd.DataFrame): phrase-by-domain frequency matrix
        threshold (float): dominance threshold (e.g., 0.9)
        export_path (str): CSV path to save removed phrases

    Returns:
        pd.DataFrame: cleaned nij_matrix with dominant phrases removed
    """
    phrase_totals = nij_matrix.sum(axis=1)
    phrase_max_per_domain = nij_matrix.max(axis=1)
    dominance_ratio = phrase_max_per_domain / phrase_totals

    too_good_phrases = dominance_ratio[dominance_ratio > threshold]

    logging.info(f"Found {len(too_good_phrases)} phrases that are too good (threshold: {threshold})")

    exported_df = nij_matrix.loc[too_good_phrases.index].copy()
    exported_df["dominance_ratio"] = too_good_phrases
    exported_df.to_csv(export_path, index_label="phrase")

    logging.info(f"Exported too good phrases to {export_path}")

    return nij_matrix.drop(index=too_good_phrases.index, errors='ignore')

def compute_mutual_information_scores(nij_matrix, export_path):
    """
    Computes mutual information scores for each phrase in a phrase-domain frequency matrix.

    The score reflects how informative each phrase is in distinguishing between domains.
    Higher scores indicate phrases that are disproportionately associated with specific domains.

    Returns:
        Sorted series of MI scores indexed by phrase, descending.
    """
    n_total = nij_matrix.values.sum()
    if n_total == 0:
        raise ValueError("The total count in nij_matrix is zero. Cannot compute probabilities.")

    # Joint probability P(i, j)
    pij_matrix = nij_matrix / n_total

    # Marginal probabilities
    pi_matrix = pij_matrix.sum(axis=1)
    pj_matrix = pij_matrix.sum(axis=0)

    scores = {}

    for phrase in pij_matrix.index:
        score = 0.0
        for domain in pij_matrix.columns:
            pij = pij_matrix.at[phrase, domain]
            if pij > 0:
                pi = pi_matrix.at[phrase]
                pj = pj_matrix.at[domain]
                score += pij * np.log2(pij / (pi * pj))
        scores[phrase] = score

    info_scores = pd.Series(scores).sort_values(ascending=False)

    info_scores.to_csv(export_path, sep="\t", header=["MI_score"], index_label="phrase")

    logging.info(f"Mutual information scores exported to {export_path}")

    return info_scores

def compute_pointwise_mutual_information(nij_matrix: pd.DataFrame, export_path: str):
    """
    Computes Pointwise Mutual Information (PMI) for each (phrase, domain) pair.

    PMI(phrase, domain) = log2( P(phrase, domain) / (P(phrase) * P(domain)) )

    Returns:
        pd.DataFrame with columns: ['phrase', 'domain', 'pmi'], sorted by PMI descending.
    """
    n_total = nij_matrix.values.sum()
    if n_total == 0:
        raise ValueError("The total count in nij_matrix is zero. Cannot compute probabilities.")

    pij_matrix = nij_matrix / n_total
    pi_matrix = pij_matrix.sum(axis=1)
    pj_matrix = pij_matrix.sum(axis=0)

    rows = []

    for phrase in pij_matrix.index:
        for domain in pij_matrix.columns:
            pij = pij_matrix.at[phrase, domain]
            if pij > 0:
                pi = pi_matrix.at[phrase]
                pj = pj_matrix.at[domain]
                pmi = np.log2(pij / (pi * pj))
                rows.append((phrase, domain, pmi))

    pmi_df = pd.DataFrame(rows, columns=["phrase", "domain", "pmi"]).sort_values(by="pmi", ascending=False)
    pmi_df.to_csv(export_path, sep="\t", index=False)

    logging.info(f"Pointwise Mutual Information scores exported to {export_path}")
    return pmi_df

if __name__ == "__main__":

    logging.info("Reading dataset...")
    df = pd.read_csv(INPUT_FILE)

    logging.info("Droping rows with missing text or source...")
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
    df['all_ngrams'] = df.apply(lambda row: row['monograms'] | row['bigrams'] | row['trigrams'], axis=1)


    logging.info("Creating phrase frequency matrix...")
    phrase_frequency_matrix = create_phrase_frequency_matrix(df)
    logging.info(f"The frequency matrix has: {len(phrase_frequency_matrix.index)} elements")

    logging.info("Removing redundant ngrams that are contained in longer ngrams...")
    monograms = set(mono for monograms_list in df['monograms'] for mono in monograms_list)
    bigrams = set(bi for bigrams_list in df['bigrams'] for bi in bigrams_list)
    trigrams = set(tri for trigrams_list in df['trigrams'] for tri in trigrams_list)

    reverse_index_bigrams = build_reverse_index_by_ngram(bigrams, 1)
    reverse_index_trigrams = build_reverse_index_by_ngram(trigrams, 1)
    reverse_index_bigrams_in_trigrams = build_reverse_index_by_ngram(trigrams, 2)

    monograms_in_bigrams_to_delete = find_phrases_to_delete_fast(monograms,
                                                                 reverse_index_bigrams,
                                                                 phrase_frequency_matrix,
                                                                 threshold=0.7)
    monograms_in_trigrams_to_delete =  find_phrases_to_delete_fast(monograms,
                                                                   reverse_index_trigrams,
                                                                   phrase_frequency_matrix,
                                                                   threshold=0.7)
    bigrams_to_delete = find_phrases_to_delete_fast(bigrams,
                                                    reverse_index_bigrams_in_trigrams,
                                                    phrase_frequency_matrix,
                                                    threshold=0.7)
    phrase_frequency_matrix = phrase_frequency_matrix.drop(index=
                                monograms_in_bigrams_to_delete +
                                monograms_in_trigrams_to_delete +
                                bigrams_to_delete,
                                errors='ignore')
    logging.info(f"The frequency matrix has: {len(phrase_frequency_matrix.index)} elements")

    logging.info("Filtering too good phrases...")
    phrase_frequency_matrix = filter_too_good_phrases(phrase_frequency_matrix,
                                                      export_path=TOO_GOOD_PHRASES_FILE,
                                                      threshold=0.9)
    logging.info(f"The frequency matrix has: {len(phrase_frequency_matrix.index)} elements")

    logging.info("Removing phrases that contain 'PERIOD'...")
    phrase_frequency_matrix = phrase_frequency_matrix[~phrase_frequency_matrix.index.str.contains("PERIOD")]
    logging.info(f"The frequency matrix has: {len(phrase_frequency_matrix.index)} elements")

    phrase_frequency_matrix.to_csv(MATRIX_FILE, index=True)
    logging.info(f"Phrase frequency matrix saved to {MATRIX_FILE}")

    # logging.info("Computing mutual information scores...")
    # mutual_info_scores = compute_mutual_information_scores(phrase_frequency_matrix,
    #                                                        export_path=MI_FILE)

    logging.info("Computing pointwise mutual information scores...")
    pmi_scores = compute_pointwise_mutual_information(phrase_frequency_matrix,
                                                      export_path=PMI_FILE)
