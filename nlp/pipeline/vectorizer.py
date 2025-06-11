import spacy
from sklearn.feature_extraction.text import CountVectorizer

class TextVectorizer(CountVectorizer):
    def __init__(
        self,
        spacy_model: str = "ro_core_news_lg",
        lemmatize: bool = False,
        ngram_range: tuple = (1, 3),
        min_df: int = 5,
        max_df: float = 0.85,
    ):
        # load SpaCy
        self.nlp       = spacy.load(spacy_model)
        self.stop_words= self.nlp.Defaults.stop_words
        self.lemmatize = lemmatize

        # delegate all vectorizer args to the parent
        super().__init__(
            tokenizer=self._tokenize,
            preprocessor=lambda x: x,
            lowercase=False,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            strip_accents=None
        )

    def _chunk_text(self, text: str, max_length: int):
        for i in range(0, len(text), max_length):
            yield text[i : i + max_length]

    def _tokenize(self, doc: str) -> list[str]:
        tokens: list[str] = []
        for piece in self._chunk_text(doc, self.nlp.max_length):
            for token in self.nlp(piece):
                if not token.is_alpha or token.lower_ in self.stop_words:
                    continue
                txt = token.lemma_ if self.lemmatize else token.text
                tokens.append(txt.lower())
        return tokens
