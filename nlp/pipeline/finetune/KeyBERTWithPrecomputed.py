from bertopic.representation._keybert import KeyBERTInspired
from typing import List, Mapping, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class KeyBERTWithPrecomputed(KeyBERTInspired):
    """
    KeyBERT-inspired that uses:
      • precomputed doc embeddings for topic centroids
      • precomputed word embeddings for candidate terms
    """
    def __init__(
        self,
        embeddings: np.ndarray,
        word_embs: np.ndarray,
        token_to_idx: Mapping[str, int],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.precomputed_embeddings = embeddings    # shape (n_docs, D)
        self.word_embs              = word_embs     # shape (vocab, D)
        self.token_to_idx           = token_to_idx
        self._mean_word_emb         = word_embs.mean(axis=0)  # fallback vector

    def _extract_embeddings(
        self,
        topic_model,
        topics: Mapping[int, List[str]],
        representative_docs: List[str],
        repr_doc_indices: List[List[int]]
    ) -> Union[np.ndarray, List[str]]:
        # 1) Topic centroids
        flat_idxs       = [i for grp in repr_doc_indices for i in grp]
        repr_embs       = self.precomputed_embeddings[flat_idxs]
        topic_embeddings = [repr_embs[idxs].mean(axis=0) for idxs in repr_doc_indices]

        # 2) Deterministic vocab: iterate topics in sorted order
        vocab = []
        for topic_id in sorted(topics):
            for word in topics[topic_id]:
                if word not in vocab:
                    vocab.append(word)

        # 3) Batch-lookup word embeddings with mean fallback
        idxs = np.array([ self.token_to_idx.get(w, -1) for w in vocab ], dtype=int)
        safe = idxs.copy()
        mask = (safe == -1)
        safe[mask] = 0
        word_embeddings = self.word_embs[safe]
        if mask.any():
            word_embeddings[mask] = self._mean_word_emb

        # 4) Similarity
        sim = cosine_similarity(topic_embeddings, word_embeddings)
        return sim, vocab
