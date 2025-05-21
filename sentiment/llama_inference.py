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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL = "/models/llama2-70b-gptq"
TEMPERATURE = 0.0
MAX_TOKENS = 10

def build_prompt(sentence: str, entity: str) -> str:
    examples = (
        "Propoziție: Klaus Iohannis a participat la o conferință în Bruxelles.\n"
        "Entitate: Klaus Iohannis\n"
        "Etichetă: neutru\n\n"
        "Propoziție: USR a fost felicitat pentru inițiativa educațională.\n"
        "Entitate: USR\n"
        "Etichetă: pozitiv\n\n"
        "Propoziție: PSD a fost acuzat de obstrucționarea votului în diaspora.\n"
        "Entitate: PSD\n"
        "Etichetă: negativ\n\n"
    )

    instruction = (
        f"Clasifică atitudinea exprimată față de entitatea „{entity}” în propoziția următoare.\n"
        f"Eticheta trebuie să fie exact una dintre:\n"
        f"- pozitiv: propoziția exprimă susținere, laudă sau apreciere\n"
        f"- negativ: propoziția exprimă critică, acuzație sau opoziție\n"
        f"- neutru: propoziția este informativă, descrie fapte sau nu exprimă o opinie clară\n\n"
        f"Important: dacă propoziția doar prezintă informații sau nu este clară, răspunde cu *neutru*.\n"
        f"Răspunsul trebuie să fie DOAR un singur cuvânt: pozitiv, negativ sau neutru.\n\n"
    )

    target = f"Propoziție: {sentence}\nEtichetă:"
    return examples + instruction + target

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
    df = pd.read_csv("dataset/romanian_political_articles_v2_ner.csv")

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

    output_file = f"dataset/romanian_political_articles_v2_sentiment_part{split_part}.csv"
    df.to_csv(output_file, index=False)
    logging.info("Stance extraction completed and saved to csv.")
