from bertopic.representation import MaximalMarginalRelevance

def create_representation_model():
    return MaximalMarginalRelevance(
        diversity=0.3
    )
