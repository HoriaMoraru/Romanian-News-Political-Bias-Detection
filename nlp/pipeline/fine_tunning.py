from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import KeyBERTInspired

from nlp.KeyBERTWithPrecomputed import KeyBERTWithPrecomputed
from nlp.MMRWithPrecomputed import MMRWithPrecomputed

def create_representation_model():
    keybert = KeyBERTInspired(
        top_n_words=30,
    )

    mmr = MaximalMarginalRelevance(
        diversity=0.5,
        top_n_words=10
    )

    return [keybert, mmr]

def create_representation_model_precomputed_embeddings(
    doc_embeddings,
    word_embeddings,
    token_to_idx
):
    keybert = KeyBERTWithPrecomputed(
        embeddings=doc_embeddings,
        top_n_words=15,
        nr_candidates=50
    )
    mmr = MMRWithPrecomputed(
        doc_embeddings=doc_embeddings,
        word_embs=word_embeddings,
        token_to_idx=token_to_idx,
        diversity=0.3,
        top_n_words=10
    )

    return [keybert, mmr]
