import re
import html

class Preprocessor:
    """
    Text Preprocessor for Romanian news article cleanup.
    Provides methods to remove unwanted patterns such as HTML tags,
    URLs, file references, quotes, and normalizes whitespace and Romanian characters.
    """

    def __init__(self, text: str):
        self.text = text or ""

    def remove_quotes(self):
        self.text = html.unescape(self.text)
        pattern = r'["\'„”‘’«»]{1,10}[^"\'„”‘’«»]{1,1000}?["\'„”‘’«»]{1,10}'
        self.text = re.sub(pattern, '', self.text, flags=re.DOTALL)
        return self

    def remove_hyperlinks(self):
        url_pattern = r'(https?://\S+|www\.\S+)'
        self.text = re.sub(url_pattern, '', self.text, flags=re.IGNORECASE)
        return self

    def remove_file_references(self):
        file_extensions = (
            r'\b\S+\.('
            r'pdf|png|jpg|jpeg|gif|svg|doc|docx|xls|xlsx|ppt|pptx|zip|rar|7z|'
            r'mp4|mp3|txt|exe'
            r')\b'
        )
        self.text = re.sub(file_extensions, '', self.text, flags=re.IGNORECASE)
        return self

    def remove_html(self):
        self.text = re.sub(r"<.*?>", "", self.text)
        return self

    def remove_digits_keep_years(self):
        self.text = re.sub(r'\b(?!20\d{2})\d+\b', '', self.text)
        return self

    def normalize_whitespace(self):
        self.text = re.sub(r'\s+', ' ', self.text).strip()
        return self

    def remove_weird_punctuation(self):
        self.text = re.sub(r'[\*\•\·@~^_`+=\\|]', '', self.text)
        return self

    def preprocess_for_romanian_models(self):
        self.text = (
            self.text.replace("ţ", "ț")
                     .replace("ş", "ș")
                     .replace("Ţ", "Ț")
                     .replace("Ş", "Ș")
        )
        return self

    def process(self) -> str:
        if not self.text.strip():
            return ""
        return (
            self.preprocess_for_romanian_models()
                .remove_html()
                .remove_hyperlinks()
                .remove_file_references()
                .remove_quotes()
                .remove_digits_keep_years()
                .remove_weird_punctuation()
                .normalize_whitespace()
                .text
        )
