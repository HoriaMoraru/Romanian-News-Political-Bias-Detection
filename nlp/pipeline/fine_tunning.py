from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import KeyBERTInspired

def create_representation_model():
    keybert = KeyBERTInspired(
        top_n_words=30,
    )

    mmr = MaximalMarginalRelevance(
        diversity=0.5,
        top_n_words=10
    )

    return [keybert, mmr]
