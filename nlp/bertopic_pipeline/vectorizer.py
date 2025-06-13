import spacy
from sklearn.feature_extraction.text import CountVectorizer

class TextVectorizer(CountVectorizer):
    def __init__(
        self,
        spacy_model: str = "ro_core_news_lg",
        lemmatize: bool = False,
        ngram_range: tuple = (1, 2),
        min_df: int = 5,
        max_df: float = 0.85,
    ):
        self.spacy_model = spacy_model
        self.lemmatize   = lemmatize
        self.ngram_range = ngram_range
        self.min_df      = min_df
        self.max_df      = max_df

        self.nlp        = spacy.load(self.spacy_model)
        self.stop_words = self.nlp.Defaults.stop_words

        super().__init__(
            tokenizer=self._tokenize,
            preprocessor=lambda x: x,
            lowercase=False,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=self.stop_words,
            strip_accents=None
        )

    def _chunk_text(self, text: str, max_length: int):
        for i in range(0, len(text), max_length):
            yield text[i : i + max_length]

    def _tokenize(self, doc: str) -> list[str]:
        tokens = []
        for piece in self._chunk_text(doc, self.nlp.max_length):
            for token in self.nlp(piece):
                if not token.is_alpha or token.lower_ in self.stop_words:
                    continue
                txt = token.lemma_ if self.lemmatize else token.text
                tokens.append(txt.lower())
        return tokens
