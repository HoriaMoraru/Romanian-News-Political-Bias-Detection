from bertopic.representation._mmr import MaximalMarginalRelevance, mmr
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List, Mapping, Tuple

class MMRWithPrecomputed(MaximalMarginalRelevance):
    """
    MMR that uses:
      • precomputed doc embeddings
      • precomputed word embeddings
    """
    def __init__(
        self,
        doc_embeddings: np.ndarray,
        word_embs: np.ndarray,
        token_to_idx: Mapping[str, int],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.doc_embeddings = doc_embeddings    # (n_docs, D)
        self.word_embs      = word_embs         # (vocab, D)
        self.token_to_idx   = token_to_idx
        self._mean_word_emb = word_embs.mean(axis=0)

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        updated = {}

        # 1) Unpack all four outputs correctly
        repr_mapping, repr_docs, repr_indices, repr_ids = \
            topic_model._extract_representative_docs(c_tf_idf, documents, topics)

        # For ordering, get sorted topic IDs
        topic_order = list(topics.keys())

        for tid, word_scores in topics.items():
            words = [w for w, _ in word_scores]

            # 2) Get doc indices for this topic
            idx_in_order = topic_order.index(tid)
            doc_idxs     = repr_indices[idx_in_order]
            if not doc_idxs:
                warnings.warn(f"No repr docs for topic {tid}; skipping MMR.")
                updated[tid] = word_scores
                continue

            topic_emb = self.doc_embeddings[doc_idxs].mean(axis=0, keepdims=True)

            # 3) Batch-lookup word embeddings
            idxs = np.array([ self.token_to_idx.get(w, -1) for w in words ], dtype=int)
            safe = idxs.copy()
            mask = (safe == -1)
            safe[mask] = 0
            word_embeddings = self.word_embs[safe]
            if mask.any():
                word_embeddings[mask] = self._mean_word_emb

            # 4) Run MMR
            picks = mmr(topic_emb,
                        word_embeddings,
                        words,
                        self.diversity,
                        self.top_n_words)

            # 5) Filter original scores
            updated[tid] = [(w, s) for w, s in word_scores if w in picks]

        return updated
