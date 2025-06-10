import spacy
from sklearn.feature_extraction.text import CountVectorizer

class TextVectorizer:
    def __init__(
        self,
        spacy_model: str = "ro_core_news_lg",
        lemmatize: bool = False,
        ngram_range: tuple = (1, 3),
        min_df: int = 5,
        max_df: float = 0.85,
    ):
        self.nlp = spacy.load(spacy_model)
        self.stop_words = self.nlp.Defaults.stop_words
        self.lemmatize = lemmatize

        self.vectorizer = CountVectorizer(
            tokenizer=self._tokenize,
            preprocessor=lambda x: x,       # skip built-in preprocessing
            lowercase=False,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            strip_accents=None
        )

    def _chunk_text(text: str, max_length: int):
        for i in range(0, len(text), max_length):
            yield text[i : i + max_length]

    def _tokenize(self, doc: str) -> list[str]:
        """
        Split into spaCy tokens (in chunks if length exceeds spaCy limits), filter,
        and optionally lemmatize.
        """
        tokens: list[str] = []
        for piece in self._chunk_text(doc, self.nlp.max_length):
            for token in self.nlp(piece):
                if not token.is_alpha or token.lower_ in self.stop_words:
                    continue
                txt = token.lemma_ if self.lemmatize else token.text
                tokens.append(txt.lower())
        return tokens

    def get_vectorizer(self) -> CountVectorizer:
        return self.vectorizer
