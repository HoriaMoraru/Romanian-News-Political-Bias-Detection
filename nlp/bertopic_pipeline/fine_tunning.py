from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import KeyBERTInspired

def create_representation_model():
    keybert = KeyBERTInspired(
        top_n_words=30,
    )

    mmr = MaximalMarginalRelevance(
        diversity=0.7,
        top_n_words=15
    )

    return [keybert, mmr]

