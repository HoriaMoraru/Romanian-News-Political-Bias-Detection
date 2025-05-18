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
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL = "/models/llama2-70b-gptq"
TEMPERATURE = 0.0
MAX_TOKENS = 20

def build_prompt(sentence: str, entity: str) -> str:
    return (
        f"Analizează următoarea propoziție și clasifică atitudinea față de entitatea „{entity}”.\n"
        f"Răspunsul trebuie să fie un singur cuvânt, exact unul dintre: pozitiv, negativ sau neutru.\n\n"
        f"Propoziție: {sentence}\n"
        f"Răspuns:"
    )

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

def analyze_stance(text: str, entities: List[str], client) -> List[tuple]:
    results = []
    sentences = sent_tokenize(text)

    for entity in entities:
        relevant_sentences = [s for s in sentences if entity.lower() in s.lower()]
        if not relevant_sentences:
            results.append((entity, "NO_ENTITY"))
            continue
        entity_context = " ".join(relevant_sentences)
        prompt = build_prompt(entity_context, entity)
        stance = query_llm(prompt, client)
        results.append((entity, stance))
    return results

def safe_eval_entities(entities_str):
    try:
        return ast.literal_eval(entities_str)
    except Exception as e:
        logging.warning(f"Could not parse entities: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"1", "2", "3", "4"}:
        print("Usage: python script.py [1|2|3|4]")
        sys.exit(1)

    split_part = int(sys.argv[1])

    logging.info("Loading OpenAI client...")
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    logging.info("Loading dataset...")
    df = pd.read_csv("dataset/romanian_political_articles_v1_ner.csv")

    logging.info("Shuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logging.info("Splitting dataset into quarters...")
    total = len(df)
    quarter_size = total // 4
    start = (split_part - 1) * quarter_size
    end = start + quarter_size if split_part < 4 else total
    df = df.iloc[start:end]

    logging.info("Processing rows...")
    tqdm.pandas(desc=f"Processing quarter {split_part} (rows {start} to {end})...")
    df['stance'] = df.progress_apply(
        lambda row: analyze_stance(row['maintext'], safe_eval_entities(row['ner']), client), axis=1
    )

    output_file = f"dataset/romanian_political_articles_v1_ner_sentiment_part{split_part}.csv"
    df.to_csv(output_file, index=False)
    logging.info("Stance extraction completed and saved to csv.")
