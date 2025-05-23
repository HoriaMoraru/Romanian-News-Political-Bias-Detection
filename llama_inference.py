import pandas as pd
from openai import OpenAI
import re
from typing import List
import logging
import time
import ast
from tqdm import tqdm
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize
import sys
from transformers import LlamaTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL = "/models/Llama-3.3-70B-Instruct-bnb-4bit"
TEMPERATURE = 0.0
MAX_TOKENS = 10
MAX_CONTEXT_TOKENS = 4096

def build_prompt(text: str, entity: str) -> str:
    return f"""Evaluează atitudinea exprimată față de entitatea „{entity}” în textul de mai jos.

                Alege exact una dintre următoarele etichete:
                - pozitiv
                - neutru
                - negativ

                Alege eticheta care reflectă cel mai bine atitudinea din text, chiar dacă este subtilă.
                Alege „neutru” decât dacă nu există nici un indiciu de atitudine pozitivă sau negativă clar si daca propozitia este pur informativa.

                Text:
                \"\"\"{text}\"\"\"

                Răspuns (doar un cuvânt):"""


def query_llm(prompt, client, max_retries=3):

    for _ in range(max_retries):
        try:
            response = client.completions.create(
                model=MODEL,
                prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            message = response.choices[0].text.strip()
            match = re.search(r"\b(pozitiv|negativ|neutru)\b", message, re.IGNORECASE)
            stance = match.group(1).upper() if match else "UNKNOWN"
            return stance
        except Exception as e:
            logging.warning(f"Error querying llm for sentiment: {e}")
            time.sleep(1)

    return "ERROR"

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))

def truncate_context_to_fit(sentences, entity, tokenizer, max_tokens=4000):
    selected = []
    for sent in sentences:
        selected.append(sent)
        tentative_context = " ".join(selected)
        prompt = build_prompt(tentative_context, entity)
        if count_tokens(prompt, tokenizer) > max_tokens:
            selected.pop()
            break
    return " ".join(selected)

def analyze_stance(text: str, entities: List[str], client, tokenizer) -> List[tuple]:
    results = []
    sentences = sent_tokenize(text)

    for entity in entities:
        relevant_sentences = [s for s in sentences if re.search(rf"\b{re.escape(entity)}\b", s, flags=re.IGNORECASE)]

        if not relevant_sentences:
            results.append((entity, "NO_ENTITY"))
            continue

        entity_context = truncate_context_to_fit(relevant_sentences,
                                                 entity,
                                                 tokenizer,
                                                 MAX_CONTEXT_TOKENS - MAX_TOKENS - 1)
        prompt = build_prompt(entity_context, entity)
        stance = query_llm(prompt, client)
        results.append({"entity": entity, "stance": stance})

    return results

def safe_eval_entities(entities_str):
    try:
        return ast.literal_eval(entities_str)
    except Exception as e:
        logging.warning(f"Could not parse entities: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {str(i) for i in range(1, 9)}:
        print("Usage: python script.py [1|2|3|4|5|6|7|8]")
        sys.exit(1)

    split_part = int(sys.argv[1])

    logging.info("Loading OpenAI client...")
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    logging.info("Loading dataset...")
    df = pd.read_csv("dataset/romanian_political_articles_v2_ner.csv")

    logging.info("Splitting dataset into 8 parts...")
    total = len(df)
    part_size = total // 8
    start = (split_part - 1) * part_size
    end = start + part_size if split_part < 8 else total
    df = df.iloc[start:end]

    logging.info("Getting tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained("/models/llama2-70b-gptq")

    logging.info("Processing rows...")
    tqdm.pandas(desc=f"[Part {split_part}] Rows {start}–{end}")
    df['stance'] = df.progress_apply(
        lambda row: analyze_stance(row['cleantext'], safe_eval_entities(row['ner']), client, tokenizer), axis=1
    )

    output_file = f"dataset/romanian_political_articles_v2_sentiment_part{split_part}.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"Stance extraction completed and saved {len(df)} rows to {output_file}.")
