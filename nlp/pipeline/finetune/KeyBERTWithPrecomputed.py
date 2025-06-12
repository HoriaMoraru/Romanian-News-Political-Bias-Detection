from bertopic.representation._keybert import KeyBERTInspired
from typing import List, Mapping, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class KeyBERTWithPrecomputed(KeyBERTInspired):
    """
    KeyBERT-inspired representation that uses precomputed document embeddings
    for topic centroids and precomputed word embeddings for candidate terms.
    """
    def __init__(self,
                 embeddings: np.ndarray,
                 word_embs: np.ndarray,
                 token_to_idx: Mapping[str, int],
                 **kwargs):
        """
        Args:
          embeddings:    (n_docs, D) array of document embeddings
          word_embs:     (vocab_size, D) array of word embeddings
          token_to_idx:  mapping token → row in word_embs
          **kwargs:      passed through to KeyBERTInspired (top_n_words, nr_candidates, etc.)
        """
        super().__init__(**kwargs)
        self.precomputed_embeddings = embeddings
        self.word_embs              = word_embs
        self.token_to_idx           = token_to_idx

    def _extract_embeddings(
        self,
        topic_model,
        topics: Mapping[int, List[str]],
        representative_docs: List[str],
        repr_doc_indices: List[List[int]]
    )-> Union[np.ndarray, List[str]]:
        flat_idxs = [i for grp in repr_doc_indices for i in grp]
        repr_embs = self.precomputed_embeddings[flat_idxs]

        topic_embeddings = [
            repr_embs[idxs].mean(axis=0)
            for idxs in repr_doc_indices
        ]

        vocab = list({w for words in topics.values() for w in words})

        word_emb_list = []
        for word in vocab:
            idx = self.token_to_idx.get(word)
            if idx is None:
                word_emb_list.append(np.zeros(self.word_embs.shape[1]))
            else:
                word_emb_list.append(self.word_embs[idx])
        word_embeddings = np.vstack(word_emb_list)

        sim = cosine_similarity(topic_embeddings, word_embeddings)
        return sim, vocab
