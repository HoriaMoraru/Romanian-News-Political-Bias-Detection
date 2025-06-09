from bertopic.representation._keybert import KeyBERTInspired
from typing import List, Mapping, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class KeyBERTWithPrecomputed(KeyBERTInspired):
    def __init__(self,
                 embeddings: np.ndarray,
                 embedding_model_checkpoint: str,
                 **kwargs):
        """
        embeddings: your (n_docs, dim) array you loaded with np.load(...)
        embedding_model_checkpoint: e.g. "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        kwargs are passed to KeyBERTInspired (top_n_words, etc.)
        """
        super().__init__(**kwargs)
        self.precomputed_embeddings = embeddings
        self.checkpoint = embedding_model_checkpoint
        self.word_model = SentenceTransformer(self.checkpoint)

    def _extract_embeddings(self,
                            topic_model,
                            topics: Mapping[int, List[str]],
                            representative_docs: List[str],
                            repr_doc_indices: List[List[int]]):
        # 1) document embeddings → use precomputed
        #   repr_doc_indices is a list of lists, but KeyBERTInspired flattens them,
        #   so repr_embeddings = embeddings[flat_indices]
        flat_idx = [i for sub in repr_doc_indices for i in sub]
        repr_embeddings = self.precomputed_embeddings[flat_idx]

        # group back into topic blocks
        topic_embeddings = [
            np.mean(
                repr_embeddings[ sum(len(sub) for sub in repr_doc_indices[:i]) :
                                 sum(len(sub) for sub in repr_doc_indices[:i+1]) ],
                axis=0
            )
            for i in range(len(repr_doc_indices))
        ]

        # 2) word embeddings → embed the vocabulary words
        vocab = list({w for words in topics.values() for w in words})
        word_embs = self.word_model.encode(vocab, convert_to_numpy=True)
        sim = cosine_similarity(topic_embeddings, word_embs)
        return sim, vocab
