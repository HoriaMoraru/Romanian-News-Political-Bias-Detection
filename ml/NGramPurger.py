from collections import defaultdict
from typing import List, Set, Dict
import pandas as pd

class NGramPurger:
    def __init__(
        self,
        longer_phrases: List[str],
        shorter_phrases: List[str],
        ngram_size: int,
        threshold: float = 0.7
    ):
        """
        Args:
          longer_phrases: List of candidate longer n-grams (e.g. trigrams).
          shorter_phrases: List of shorter n-grams to potentially purge (e.g. bigrams).
          ngram_size: The size of the shorter phrases you want to check (e.g. 2 for bigrams).
          threshold: If a longer phrase occurs at least `threshold * freq(shorter)`,
                     the shorter phrase is considered redundant.
        """
        self.longer_phrases  = longer_phrases
        self.shorter_phrases = shorter_phrases
        self.ngram_size      = ngram_size
        self.threshold       = threshold

        self._reverse_index = self._build_reverse_index()

    def _build_reverse_index(self) -> Dict[str, List[str]]:
        """Maps each shorter subphrase → list of longer phrases that contain it."""
        rev: Dict[str, List[str]] = defaultdict(list)
        for phrase in self.longer_phrases:
            tokens = phrase.split()
            for i in range(len(tokens) - self.ngram_size + 1):
                sub = " ".join(tokens[i:i + self.ngram_size])
                rev[sub].append(phrase)
        return rev

    def find_redundant(
        self,
        freq_matrix: pd.DataFrame
    ) -> List[str]:
        """
        Given a phrase x domain frequency matrix (rows=indexed by phrase, columns by source/domain),
        returns the subset of `shorter_phrases` that are “subsumed” by any longer phrase.

        freq_matrix.loc[phrase].sum() gives total frequency of that phrase.
        """
        to_delete = []
        for short in self.shorter_phrases:
            if short not in freq_matrix.index:
                continue
            short_count = freq_matrix.loc[short].sum()
            for long in self._reverse_index.get(short, []):
                if long not in freq_matrix.index:
                    continue
                long_count = freq_matrix.loc[long].sum()
                if long_count >= self.threshold * short_count:
                    to_delete.append(short)
                    break
        return to_delete
