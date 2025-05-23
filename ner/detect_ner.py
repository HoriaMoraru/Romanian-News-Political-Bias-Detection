import pandas as pd
import re
import logging
import html
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, pipeline
from ..preprocessing.Preprocessor import Preprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILE = "dataset/romanian_political_articles_v2_shuffled.csv"
OUTPUT_FILE = "dataset/romanian_political_articles_v2_ner.csv"

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

def extract_named_entities(text, nlp_ro, ner_pipeline):
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

if __name__ == "__main__":
    logging.info("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["maintext", "source_domain"])

    logging.info("Loading Romanian spaCy model...")
    nlp_ro = spacy.load("ro_core_news_sm")

    logging.info("Preprocessing text...")
    tqdm.pandas()
    df['cleantext'] = df['maintext'].progress_apply(lambda text: Preprocessor(text).process())
    df = df[df['cleantext'].str.split().str.len() > 30]
    logging.info(f"Filtered dataset size: {len(df)}")

    logging.info("Loading NER model...")
    ner_pipeline = get_romanian_ner_nlp_pipeline()

    logging.info("Extracting named entities...")
    df['ner'] = df['cleantext'].progress_apply(lambda text: extract_named_entities(text, nlp_ro, ner_pipeline))

    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Saved NER output to {OUTPUT_FILE}")
