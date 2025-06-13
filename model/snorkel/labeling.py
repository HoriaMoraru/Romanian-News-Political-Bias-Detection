"""
labeling_functions.py

This script defines a set of Snorkel labeling functions (LFs) to weakly label news articles
as liberal or conservative (or abstain), using only data-driven signals and the two pre-existing
500-word lexica for liberal and conservative terms, plus two pre-trained clustering models:
    1. phrase_cluster_model: clusters domains (by their phrase-based features) into “liberal” or “conservative”
    2. entity_sent_cluster_model: clusters per-article entity-sentiment vectors into “liberal” or “conservative”

This script uses Snorkel v0.9 API: https://snorkel.readthedocs.io/en/master/packages/snorkel-labeling.html
"""

import re
import pickle
import pandas as pd
from urllib.parse import urlparse

from snorkel.labeling import labeling_function, PandasLFApplier

# ───────────────────────────────────────────────────────────────────────────────
# 1. LABEL ENUMERATIONS
# ───────────────────────────────────────────────────────────────────────────────

ABSTAIN      = -1
LIBERAL      = 0
CONSERVATIVE = 1

# ───────────────────────────────────────────────────────────────────────────────
# 2. LOAD LEFT/RIGHT LEXICA (DATA‐DRIVEN LISTS)
# ───────────────────────────────────────────────────────────────────────────────

CONSERVATIVE_LEXICON_PATH = "dataset/snorkel/conservative.csv"
LIBERAL_LEXICON_PATH      = "dataset/snorkel/liberal.csv"

def load_lexicon(path):
    df = pd.read_csv(path, header=None, names=["word"])
    return set(df["word"].astype(str).str.lower().tolist())

CONSERVATIVE_WORDS = load_lexicon(CONSERVATIVE_LEXICON_PATH)
LIBERAL_WORDS      = load_lexicon(LIBERAL_LEXICON_PATH)

# ───────────────────────────────────────────────────────────────────────────────
# 3. LOAD PRE‐TRAINED CLUSTERING MODELS
# ───────────────────────────────────────────────────────────────────────────────

# These .pkl files should contain scikit‐learn‐compatible clustering objects with a .predict() method.
PHRASE_CLUSTER_MODEL_PATH        = "dataset/snorkel/phrase_cluster_model.pkl"
ENTITY_SENT_CLUSTER_MODEL_PATH   = "dataset/snorkel/entity_sent_cluster_model.pkl"

with open(PHRASE_CLUSTER_MODEL_PATH, "rb") as f:
    phrase_cluster_model = pickle.load(f)

with open(ENTITY_SENT_CLUSTER_MODEL_PATH, "rb") as f:
    entity_sent_cluster_model = pickle.load(f)

# ───────────────────────────────────────────────────────────────────────────────
# 4. MAP CLUSTER LABELS → LIBERAL/CONSERVATIVE
# ───────────────────────────────────────────────────────────────────────────────
PHRASE_CLUSTER_LABEL_TO_CLASS = {
    0: LIBERAL,
    1: CONSERVATIVE,
}

ENTITY_CLUSTER_LABEL_TO_CLASS = {
    0: LIBERAL,
    1: CONSERVATIVE,
}

# ───────────────────────────────────────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Lowercase, remove punctuation (except keep whitespace), return cleaned text.
    """
    text = text.lower()
    # Replace any non‐alphanumeric character with space
    return re.sub(r"[^a-z0-9\s]", " ", text)

def count_political_terms(text: str) -> tuple[int, int]:
    """
    Count occurrences of conservative vs. liberal words in `text`.
    Returns a tuple (liberal_count, conservative_count).
    Word matching is exact on whitespace‐tokenized words.
    """
    tokens = normalize_text(text).split()
    lib_count = 0
    con_count = 0
    for tok in tokens:
        if tok in LIBERAL_WORDS:
            lib_count += 1
        if tok in CONSERVATIVE_WORDS:
            con_count += 1
    return lib_count, con_count

def extract_domain(source_url: str) -> str:
    try:
        parsed = urlparse(source_url)
        domain = parsed.netloc

        return domain.replace("www.", "")
    except:
        return source_url.strip().lower()

def get_entity_sentiment_vector(entities: list) -> list:
    """
    Given a list of dicts with keys { "name": str, "sentiment": float },
    return a flat list (vector) of all sentiment floats, sorted by entity name.
    If an entity lacks a “sentiment” field, treat its sentiment as 0.0.
    """
    if not isinstance(entities, list) or len(entities) == 0:
        return []
    # Sort by entity name to have deterministic order
    sorted_ents = sorted(entities, key=lambda d: d.get("name", ""))
    return [ent.get("sentiment", 0.0) for ent in sorted_ents]

# ───────────────────────────────────────────────────────────────────────────────
# 6. LABELING FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

@labeling_function(pre=[lambda x: x])
def lf_political_word_balance(row) -> int:
    """
    If the text contains strictly more liberal words than conservative → LIBERAL.
    If strictly more conservative than liberal → CONSERVATIVE.
    Otherwise → ABSTAIN.
    """
    text = row.text if isinstance(row.text, str) else ""
    lib_count, con_count = count_political_terms(text)
    if lib_count > con_count:
        return LIBERAL
    if con_count > lib_count:
        return CONSERVATIVE
    return ABSTAIN


@labeling_function(pre=[lambda x: x])
def lf_domain_phrase_cluster(row) -> int:
    """
    Apply the phrase‐based cluster model on the domain extracted from source.
    If the predicted cluster maps to LIBERAL or CONSERVATIVE, return that.
    Otherwise, ABSTAIN (unlikely, but for safety).
    """
    source = row.source if isinstance(row.source, str) else ""
    domain = extract_domain(source)
    try:
        cluster_label = phrase_cluster_model.predict([domain])[0]
        return PHRASE_CLUSTER_LABEL_TO_CLASS.get(cluster_label, ABSTAIN)
    except:
        return ABSTAIN


@labeling_function(pre=[lambda x: x])
def lf_domain_entity_cluster(row) -> int:
    """
    Apply the entity‐sentiment‐based cluster model on the domain’s aggregate sentiment signature.
    We summarize each article by the vector of entity sentiments (sorted by entity name).
    Then ask the cluster model to assign a cluster, and map that to LIBERAL/CONSERVATIVE.
    """
    entities = row.entities if isinstance(row.entities, list) else []
    sent_vector = get_entity_sentiment_vector(entities)
    if len(sent_vector) == 0:
        return ABSTAIN
    try:
        cluster_label = entity_sent_cluster_model.predict([sent_vector])[0]
        return ENTITY_CLUSTER_LABEL_TO_CLASS.get(cluster_label, ABSTAIN)
    except:
        return ABSTAIN


@labeling_function(pre=[lambda x: x])
def lf_entity_sentiment_majority(row) -> int:
    """
    Compute average sentiment per unique entity name; then compare aggregate sentiment towards
    entities whose names appear lexically similar to liberal vs. conservative terms.
    WARNING: this is a weak heuristic—uses the existing lexica to guess an entity's affiliation,
    then measures sentiment. If the average sentiment towards 'liberal‐affiliated entities'
    > than that towards 'conservative‐affiliated entities', label accordingly.
    Otherwise, ABSTAIN.
    """
    entities = row.entities if isinstance(row.entities, list) else []
    # Map entity → list of sentiment scores
    lib_sents = []
    con_sents = []
    for ent in entities:
        name = ent.get("name", "").lower()
        sentiment = float(ent.get("sentiment", 0.0))
        # Check if any token in the entity name appears in one of the lexica:
        # (This relies on the 500‐word lexica but NOT on any custom lexicon beyond that.)
        name_tokens = re.sub(r"[^a-z0-9\s]", " ", name).split()
        is_lib = any(tok in LIBERAL_WORDS for tok in name_tokens)
        is_con = any(tok in CONSERVATIVE_WORDS for tok in name_tokens)
        if is_lib and not is_con:
            lib_sents.append(sentiment)
        elif is_con and not is_lib:
            con_sents.append(sentiment)
        # If an entity appears in both lexica (rare) or neither, we skip it.
    if not lib_sents and not con_sents:
        return ABSTAIN
    avg_lib_sent = sum(lib_sents) / len(lib_sents) if lib_sents else 0.0
    avg_con_sent = sum(con_sents) / len(con_sents) if con_sents else 0.0
    # If the article is markedly more positive about liberal‐affiliated entities, label LIBERAL
    # If more positive about conservative entities, label CONSERVATIVE
    if avg_lib_sent - avg_con_sent > 0.2:
        return LIBERAL
    if avg_con_sent - avg_lib_sent > 0.2:
        return CONSERVATIVE
    return ABSTAIN


@labeling_function(pre=[lambda x: x])
def lf_high_link_diversity(row) -> int:
    """
    If an article links predominantly to a single type of ideological source (liberal or conservative),
    we skip. But if an article's hyperlinks (in row.source_html) show a high ratio to liberal vs. conservative,
    label accordingly. This LF requires that row.source_html exists and contains the HTML of the article.
    """
    html = row.source_html if hasattr(row, "source_html") else ""
    if not isinstance(html, str) or html.strip() == "":
        return ABSTAIN
    # Extract all href targets:
    links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html)
    if not links:
        return ABSTAIN
    # Suppose we have a domain->bias mapping; for simplicity, check if domain contains known lexicon words:
    lib_link_count = 0
    con_link_count = 0
    for link in links:
        domain = extract_domain(link)
        # If the domain string contains any liberal word (from our lexicon), count it as liberal‐biased link
        if any(tok in domain for tok in LIBERAL_WORDS):
            lib_link_count += 1
        if any(tok in domain for tok in CONSERVATIVE_WORDS):
            con_link_count += 1
    if lib_link_count + con_link_count == 0:
        return ABSTAIN
    if lib_link_count > 2 * con_link_count:
        return LIBERAL
    if con_link_count > 2 * lib_link_count:
        return CONSERVATIVE
    return ABSTAIN


@labeling_function(pre=[lambda x: x])
def lf_text_length_extremes(row) -> int:
    """
    Extremely short or extremely long articles tend to come from fringe sites.
    As a weak heuristic, if an article has fewer than 50 tokens or more than 2500 tokens,
    default to ABSTAIN (we don't trust our other signals there). Otherwise, abstain.
    This is mostly a catch‐all; always abstains. Included as an example of a data‐driven threshold.
    """
    text = row.text if isinstance(row.text, str) else ""
    token_count = len(normalize_text(text).split())
    if token_count < 50 or token_count > 2500:
        return ABSTAIN
    return ABSTAIN  # Always abstains; placeholder for any length‐based rule you might add later.


# ───────────────────────────────────────────────────────────────────────────────
# 7. AGGREGATE LIST OF ALL LFs
# ───────────────────────────────────────────────────────────────────────────────

# You can import these into your Snorkel pipeline. For example:
#    from labeling_functions import labeling_functions
#    applier = PandasLFApplier(lfs=labeling_functions)
#    L_train = applier.apply(df_train)

labeling_functions = [
    lf_political_word_balance,
    lf_domain_phrase_cluster,
    lf_domain_entity_cluster,
    lf_entity_sentiment_majority,
    lf_high_link_diversity,
    lf_text_length_extremes,
]

# ───────────────────────────────────────────────────────────────────────────────
# 8. EXAMPLE: APPLYING LFs TO A DATAFRAME
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example usage: assume you have a CSV “articles.csv” with columns:
    #   text, source, entities (as JSON string), source_html (optional)
    import json
    from snorkel.labeling import LFAnalysis

    # Load your articles
    df = pd.read_csv("articles.csv")  # adjust path as needed

    # Parse the “entities” JSON string into actual Python lists
    def parse_entities(e):
        try:
            parsed = json.loads(e)
            if isinstance(parsed, list):
                return parsed
            return []
        except:
            return []

    if "entities" in df.columns:
        df["entities"] = df["entities"].apply(parse_entities)
    else:
        df["entities"] = [[] for _ in range(len(df))]

    # If you have “source_html” column, leave it as is (strings). Otherwise, add empty strings:
    if "source_html" not in df.columns:
        df["source_html"] = [""] * len(df)

    # Apply the LFs
    applier = PandasLFApplier(lfs=labeling_functions)
    L_train = applier.apply(df)

    # Analyze LF coverage & conflicts
    analysis = LFAnalysis(L=L_train, lfs=labeling_functions).lf_summary()
    print(analysis)  # shows coverage, overlaps, etc.

    # Save the label matrix for later
    pd.DataFrame(L_train, columns=[lf.name for lf in labeling_functions]).to_csv(
        "label_matrix.csv", index=False
    )

    print("Labeling functions applied; label matrix saved to label_matrix.csv")
