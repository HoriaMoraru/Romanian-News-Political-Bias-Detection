from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import KeyBERTInspired

from nlp.pipeline.finetune.KeyBERTWithPrecomputed import KeyBERTWithPrecomputed
from nlp.pipeline.finetune.MMRWithPrecomputed import MMRWithPrecomputed

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
        word_embs=word_embeddings,
        token_to_idx=token_to_idx,
        top_n_words=30
    )
    mmr = MMRWithPrecomputed(
        doc_embeddings=doc_embeddings,
        word_embs=word_embeddings,
        token_to_idx=token_to_idx,
        diversity=0.5
    )

    return [keybert, mmr]
