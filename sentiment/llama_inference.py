import pandas as pd
from openai import OpenAI
import re
from typing import List
import logging
import time
import ast
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading OpenAI client...")
client = OpenAI(
    api_key="EMPTY",  # Required by OpenAI client, even if unused
    base_url="http://localhost:8000/v1"
)

logging.info("Loading dataset...")
df = pd.read_csv("dataset/romanian_political_articles_v1_ner.csv")

def build_prompt(sentence: str, entity: str) -> str:
    return (
        f"Analizează următoarea propoziție și clasifică atitudinea față de entitatea „{entity}”.\n"
        f"Răspunsul trebuie să fie un singur cuvânt, exact unul dintre: pozitiv, negativ sau neutru.\n\n"
        f"Propoziție: {sentence}\n"
        f"Răspuns:"
    )

MODEL = "/models/llama2-70b-gptq"
TEMPERATURE = 0.0
MAX_TOKENS = 20

def query_llm(prompt, max_retries=3):

    for _ in range(max_retries):
        try:
            response = client.completions.create(
                model=MODEL,
                prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            message = response.choices[0].text.strip()
            logging.info(f"LLM response: {message}")
            match = re.search(r"\b(pozitiv|negativ|neutru)\b", message, re.IGNORECASE)
            stance = match.group(1).upper() if match else "UNKNOWN"
            return stance
        except Exception as e:
            logging.warning(f"Error querying llm for sentiment: {e}")
            time.sleep(1)

    return "ERROR"

def analyze_stance(text: str, entities: List[str]) -> List[tuple]:
    results = []
    for entity in entities:
        prompt = build_prompt(text, entity)
        stance = query_llm(prompt)
        results.append((entity, stance))
    return results

def safe_eval_entities(entities_str):
    try:
        return ast.literal_eval(entities_str)
    except Exception as e:
        logging.warning(f"Could not parse entities: {e}")
        return []

logging.info("Extracting stance for entities...")
tqdm.pandas(desc="Processing rows...")
# df['stance'] = df.progress_apply(
#     lambda row: analyze_stance(row['maintext'], safe_eval_entities(row['ner'])), axis=1
# )

row = df.iloc[0]
text = row["maintext"]
entities = safe_eval_entities(row["ner"])

print("Entities:", entities)
stances = analyze_stance(text, entities)
print("Stances:", stances)

# df.to_csv("dataset/romanian_political_articles_v1_ner_sentiment.csv", index=False)
# logging.info("Stance extraction completed and saved to csv.")
