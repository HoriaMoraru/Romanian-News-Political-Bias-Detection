from bertopic.vectorizers import ClassTfidfTransformer

def create_tfidf() -> ClassTfidfTransformer:
    return ClassTfidfTransformer(
        reduce_frequent_words=True
    )
