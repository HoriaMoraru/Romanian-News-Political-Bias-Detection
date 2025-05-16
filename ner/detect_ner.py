import pandas as pd
import re
import logging
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading dataset...")
input_path = "workspace/dataset/romanian_political_articles_v1.csv"
df =pd.read_csv(input_path)
df = df.dropna(subset=["maintext", "source_domain"])

logging.info("Loading Romanian Spacy model...")
nlp_ro = spacy.load("ro_core_news_sm")

def remove_quotes(text):
    return re.sub(r'[\'"„”][^\'"„”]{1,300}?[\'"„”]', '', text)

def remove_html(text):
    return re.sub(r"<.*?>", "", text)

def remove_digits_keep_years(text):
    return re.sub(r'\b(?!20\d{2})\d+\b', '', text)

def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def remove_weird_punctuation(text):
    return re.sub(r'[\*\•\·@~^_`+=\\|]', '', text)

def preprocess_for_romanian_models(text):
    return text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")

def preprocess_text(text : str):
    if not text or len(text.strip()) == 0:
        return ""

    text = preprocess_for_romanian_models(text)

    text = remove_html(text)

    text = remove_quotes(text)

    text = remove_urls(text)

    text = remove_digits_keep_years(text)

    text = remove_weird_punctuation(text)

    text = normalize_whitespace(text)

    return text.strip()

logging.info("Preprocessing text...")
tqdm.pandas()
df['cleantext'] = df['maintext'].progress_apply(preprocess_text)
df = df[df['cleantext'].str.split().str.len() > 100]

logging.info("Loading NER pipeline...")
def get_romanian_ner_nlp_pipeline():
    romanian_ner_model = "dumitrescustefan/bert-base-romanian-ner"
    tokenizer = AutoTokenizer.from_pretrained(romanian_ner_model, model_max_length=512)

    config = AutoConfig.from_pretrained(romanian_ner_model)
    config.id2label = {
        0: 'O', 1: 'B-PERSON', 2: 'I-PERSON', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-GPE', 6: 'I-GPE', 7: 'B-LOC',
        8: 'I-LOC', 9: 'B-NAT_REL_POL', 10: 'I-NAT_REL_POL', 11: 'B-EVENT', 12: 'I-EVENT', 13: 'B-LANGUAGE',
        14: 'I-LANGUAGE', 15: 'B-WORK_OF_ART', 16: 'I-WORK_OF_ART', 17: 'B-DATETIME', 18: 'I-DATETIME',
        19: 'B-PERIOD', 20: 'I-PERIOD', 21: 'B-MONEY', 22: 'I-MONEY', 23: 'B-QUANTITY', 24: 'I-QUANTITY',
        25: 'B-NUMERIC', 26: 'I-NUMERIC', 27: 'B-ORDINAL', 28: 'I-ORDINAL', 29: 'B-FACILITY', 30: 'I-FACILITY'
    }

    model = AutoModelForTokenClassification.from_pretrained(romanian_ner_model, config=config)

    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy='simple')

ner_pipeline = get_romanian_ner_nlp_pipeline()

def is_pronoun(word, doc):
    return any(token.text == word and token.pos_ == "PRON" for token in doc)

def chunk_sentences(doc, tokenizer, max_tokens=512):
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in doc.sents:
        tokens = tokenizer.tokenize(sent.text)
        if current_len + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sent.text)
        current_len += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def extract_named_entities(text):
    entities = set()

    try:
        doc = nlp_ro(text)
        chunks = chunk_sentences(doc, ner_pipeline.tokenizer)

        results = ner_pipeline(chunks, batch_size=8)

        for result in results:
            for entity in result:
                entity_group = entity['entity_group']
                word = entity['word'].strip(" .,:;\"'?!/><)(*&^%$@+-=_-”„“").replace("##", "")

                if entity_group in ['PERSON', 'GPE'] and not is_pronoun(word, doc):
                    entities.add(word)

    except Exception as e:
        logging.warning(f"NER failed: {e}")

    return sorted(entities)

logging.info("Extracting named entities...")
df['ner'] = df['cleantext'].progress_apply(extract_named_entities)

output_path = "workspace/dataset/romanian_political_articles_v1_ner.csv"
df.to_csv(output_path, index=False)
logging.info(f"Saved NER output to {output_path}")

