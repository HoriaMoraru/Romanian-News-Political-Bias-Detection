from bertopic.representation._mmr import MaximalMarginalRelevance, mmr
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List, Mapping, Tuple

class MMRWithPrecomputed(MaximalMarginalRelevance):
    """
    Extend the original MMR so that embeddings come from precomputed arrays:
      - doc_embeddings: np.ndarray, shape (n_docs, dim)
      - word_embs:      np.ndarray, shape (vocab_size, dim)
      - token_to_idx:   Dict[str,int], mapping each token → row in word_embs
    """
    def __init__(
        self,
        doc_embeddings: np.ndarray,
        word_embs: np.ndarray,
        token_to_idx: Mapping[str, int],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.doc_embeddings = doc_embeddings
        self.word_embs      = word_embs
        self.token_to_idx   = token_to_idx

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """
        For each topic:
          1. Build topic embedding from doc_embeddings (mean of repr docs)
          2. Lookup each candidate word in word_embs via token_to_idx
          3. Call inherited mmr(...) to pick the top_n_words diversified
          4. Return the filtered (word, score) list
        """
        updated_topics = {}
        repr_docs = topic_model._extract_representative_docs(c_tf_idf, documents, topics)

        for topic, topic_words in topics.items():
            words = [word[0] for word in topic_words]

            doc_idxs = repr_docs.get(topic, [])
            if not doc_idxs:
                warnings.warn(f"No representative docs for topic {topic}, skipping MMR.")
                updated_topics[topic] = topic_words
                continue

            topic_embedding = self.doc_embeddings[doc_idxs].mean(axis=0, keepdims=True)

            word_emb_list = []
            for word in words:
                idx = self.token_to_idx.get(word)
                if idx is None:
                    word_emb_list.append(np.zeros(self.word_embs.shape[1]))
                else:
                    word_emb_list.append(self.word_embs[idx])
            word_embeddings = np.vstack(word_emb_list)

            topic_words = mmr(
                topic_embedding,
                word_embeddings,
                words,
                self.diversity,
                self.top_n_words,
            )
            updated_topics[topic] = [(word, value) for word, value in topics[topic] if word in topic_words]

        return updated_topics
