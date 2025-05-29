import pandas as pd
import logging
import json
from openai import OpenAI
from tqdm import tqdm
import re
import textwrap

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL = "/models/Llama-3.3-70B-Instruct-bnb-4bit"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
BATCH_SIZE = 30

INPUT_FILE = "dataset/romanian_political_articles_v2_ner.csv"
NORMALIZED_ENTITIES_FILE = "dataset/ml/normalized_entities.json"
NORMALIZED_ENTITIES_DATASET_FILE = "dataset/romanian_political_articles_v2_ner_normalized.csv"

def build_prompt(entities: list[str]) -> str:
    examples = '\n'.join([
        '- "dl Ciolacu"',
        '- "Firea Gabriela"',
        '- "Iohannis K."',
    ])
    expected = textwrap.dedent("""\
        {
        "dl Ciolacu": "Marcel Ciolacu",
        "Firea Gabriela": "Gabriela Firea",
        "Iohannis K.": "Klaus Iohannis"
        }
    """)
    entity_list = '\n'.join([f'- "{e}"' for e in entities])
    return textwrap.dedent(f"""\
        Normalizează următoarele entități numite din limba română la forma lor canonică.
        Folosește numele complet dacă este posibil (ex: „Marcel Ciolacu” în loc de „dl Ciolacu”).
        Răspunsul trebuie să fie strict un obiect JSON, fără explicații.

        Exemplu:
        Entități:
        {examples}

        Răspuns:
        {expected}

        Acum, normalizează aceste entități:
        {entity_list}

        Răspuns:
    """)

def query_llm(prompt: str, client, max_retries=3) -> dict:
    for _ in range(max_retries):
        try:
            response = client.completions.create(
                model=MODEL,
                prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            message = response.choices[0].text.strip()

            match = re.search(r'\{[\s\S]*\}', message)
            if match:
                cleaned = match.group().strip().strip("```json").strip("```")
                logging.info(f"Raw response preview: {cleaned}...")
                return json.JSONDecoder().raw_decode(cleaned)[0]
            else:
                logging.warning("No JSON object found in response.")
        except Exception as e:
            logging.warning(f"Error querying LLM for normalization: {e}")
    return {}

def batch_entities(entities, batch_size):
    for i in range(0, len(entities), batch_size):
        yield entities[i:i + batch_size]

if __name__ == "__main__":
    logging.info("Loading OpenAI client...")
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    logging.info("Reading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df["ner"] = df["ner"].apply(eval)

    all_entities = [entity for sublist in df["ner"] for entity in sublist]
    unique_entities = list(set(all_entities))
    logging.info(f"Found {len(unique_entities)} unique entities.")

    logging.info("Normalizing entities...")
    normalized_entities = {}
    for batch in tqdm(batch_entities(unique_entities, BATCH_SIZE), desc="Normalizing"):
        prompt = build_prompt(batch)
        logging.info(f"Prompt: {prompt}")
        result = query_llm(prompt, client)
        if not result:
            logging.warning("Received empty response from LLM, skipping batch.")
            continue
        logging.info(f"Normalized {len(result)} entities in batch.")
        normalized_entities.update(result)

    with open(NORMALIZED_ENTITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized_entities, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved {len(normalized_entities)} normalized entities to {NORMALIZED_ENTITIES_FILE}")

    logging.info("Applying normalization to dataset...")
    df["ner"] = df["ner"].apply(lambda ents: [normalized_entities.get(ent, ent) for ent in ents])
    df.to_csv(NORMALIZED_ENTITIES_DATASET_FILE, index=False)
    logging.info(f"Normalized dataset saved to {NORMALIZED_ENTITIES_DATASET_FILE}")
