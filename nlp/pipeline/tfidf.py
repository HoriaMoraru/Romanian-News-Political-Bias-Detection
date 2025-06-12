from bertopic.vectorizers import ClassTfidfTransformer

def create_tfidf() -> ClassTfidfTransformer:
    return ClassTfidfTransformer(
        bm25_weighting=False,
        reduce_frequent_words=True
    )
