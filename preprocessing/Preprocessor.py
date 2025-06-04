import re
import html
import spacy

BOILERPLATE_PATTERNS_FILE = "ml/phrase_domain_frequency/utils/boilerplate_patterns.txt"

class Preprocessor:
    """
    Text Preprocessor for Romanian news article cleanup.
    Provides methods to remove unwanted patterns such as HTML tags,
    URLs, file references, quotes, and normalizes whitespace and Romanian characters.
    """
    def __init__(self, boilerplate_patterns_file: str = BOILERPLATE_PATTERNS_FILE, nlp=None):
        self.boilerplate = self._load_boilerplate_patterns(boilerplate_patterns_file)
        self.nlp = spacy.load("ro_core_news_sm") if nlp is None else nlp

    def _load_boilerplate_patterns(self, path: str) -> list[str]:
        patterns: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                literal_pattern = re.escape(line)
                patterns.append(literal_pattern)
        return patterns

    def _strip_boilerplate(self, text: str) -> str:
        for pattern in self.boilerplate:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text

    def _preprocess_for_romanian_models(self, text: str) -> str:
        return (
            text.replace("ţ", "ț")
                .replace("ş", "ș")
                .replace("Ţ", "Ț")
                .replace("Ş", "Ș")
        )

    def _remove_html(self, text: str) -> str:
        return re.sub(r"<.*?>", "", text)

    def _remove_hyperlinks(self, text: str) -> str:
        url_pattern = r"(https?://\S+|www\.\S+)"
        return re.sub(url_pattern, "", text, flags=re.IGNORECASE)

    def _remove_file_references(self, text: str) -> str:
        file_extensions = (
            r"\b\S+\.(?:"
            r"pdf|png|jpg|jpeg|gif|svg|doc|docx|xls|xlsx|ppt|pptx|zip|rar|7z|"
            r"mp4|mp3|txt|exe"
            r")\b"
        )
        return re.sub(file_extensions, "", text, flags=re.IGNORECASE)

    def _remove_quotes(self, text: str) -> str:
        text = html.unescape(text)
        pattern = r'["\'„”‘’«»]{1,10}[^"\'„”‘’«»]{1,1000}?["\'„”‘’«»]{1,10}'
        return re.sub(pattern, "", text, flags=re.DOTALL)

    def _remove_digits_keep_years(self, text: str) -> str:
        return re.sub(r"\b(?!20\d{2})\d+\b", "", text)

    def _remove_weird_punctuation(self, text: str) -> str:
        return re.sub(r"[\*\•\·@~^_`+=\\|]", "", text)

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _ml_specific(self, text: str) -> str:
        """
        Removes all punctuation (preserving only word chars and whitespace),
        lowercases each sentence’s first token,
        and joins sentences with ' PERIOD '.
        """
        doc = self.nlp(text)
        processed_sentences = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Remove punctuation except letters/digits/underscore/whitespace
            clean_sent = re.sub(r"[^\w\s]", "", sent_text)
            words = clean_sent.split()
            if not words:
                continue

            words[0] = words[0].lower()
            processed_sentences.append(" ".join(words))

        return " PERIOD ".join(processed_sentences)

    def process_nlp(self, text: str) -> str:
        """
        Applies the standard sequence of cleanup steps and returns a cleaned string:
          1. strip boilerplate
          2. normalize Romanian chars
          3. remove HTML tags
          4. remove URLs
          5. remove file references
          6. remove quoted spans
          7. remove digits (except years)
          8. remove weird punctuation
          9. collapse whitespace
        """
        if not text or not text.strip():
            return ""

        t = text
        t = self._strip_boilerplate(t)
        t = self._preprocess_for_romanian_models(t)
        t = self._remove_html(t)
        t = self._remove_hyperlinks(t)
        t = self._remove_file_references(t)
        t = self._remove_quotes(t)
        t = self._remove_digits_keep_years(t)
        t = self._remove_weird_punctuation(t)
        t = self._normalize_whitespace(t)
        return t

    def process_ml(self, text: str) -> str:
        """
        Applies the same cleanup as process_nlp(), then passes the result to _ml_specific()
        to produce an ML-ready string with tokenized sentences joined by 'PERIOD'.
        """
        cleaned = self.process_nlp(text)
        if not cleaned:
            return ""
        return self._ml_specific(cleaned)
