from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import KeyBERTInspired
from bertopic.representation import PartOfSpeech

def create_representation_model():
    keybert = KeyBERTInspired(
        top_n_words=30,
    )

    mmr = MaximalMarginalRelevance(
        diversity=0.7,
        top_n_words=15
    )

    pos=PartOfSpeech(
        'ro_core_news_lg'
    )

    return [keybert, mmr, pos]

