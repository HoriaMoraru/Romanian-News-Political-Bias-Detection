import pandas as pd
import logging
import json
import ast
from openai import OpenAI
from tqdm import tqdm
import re
import textwrap
from math import ceil
from time import sleep

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL = "/models/Llama-3.3-70B-Instruct-bnb-4bit"
TEMPERATURE = 0.0
MAX_TOKENS = 6000
BATCH_SIZE = 300

INPUT_FILE = "dataset/romanian_political_articles_v2_ner.csv"
NORMALIZED_ENTITIES_FILE = "dataset/ml/normalized_entities.json"
NORMALIZED_ENTITIES_DATASET_FILE = "dataset/romanian_political_articles_v2_ner_normalized.csv"

def build_prompt(entities: list[str]) -> str:
    examples = '\n'.join([
        '- "dl Ciolacu"',
        '- "Firea Gabriela"',
        '- "Iohannis K."',
        '- "liderii"',
        '- "Aneimariei Gavrila"',
    ])
    expected = textwrap.dedent("""\
        {
        "dl Ciolacu": "Marcel Ciolacu",
        "Firea Gabriela": "Gabriela Firea",
        "Iohannis K.": "Klaus Iohannis",
        "liderii": "lider",
        "Aneimariei Gavrila": "Anamaria Gavrila"
        }
    """)
    entity_list = '\n'.join([f'- "{e}"' for e in entities])
    return textwrap.dedent(f"""\

        Normalizează următoarele entități din limba română la forma lor canonică.

        Instrucțiuni:
        - Dacă entitatea este un nume propriu, folosește forma completă, corectă (ex: „Marcel Ciolacu” în loc de „dl Ciolacu”).
        - Dacă entitatea este un substantiv comun (ex: „liderii”), oferă forma de bază (ex: „lider”).
        - Corectează eventualele greșeli de tastare, declinări sau inversări de prenume/nume.
        - Răspunsul trebuie să fie un **obiect JSON valid**, fără explicații, fără text suplimentar.

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
                # stop=["```", "\n\n", "The final answer is", "$"]
            )
            message = response.choices[0].text.strip()

            # Try all candidate JSON blocks
            matches = re.findall(r'\{[\s\S]*?\}', message)
            for raw in sorted(matches, key=len, reverse=True):
                try:
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
                    parsed = json.loads(cleaned)
                    return {k.strip(): v.strip() for k, v in parsed.items()}
                except Exception as e:
                    continue
            logging.warning("No valid JSON block found in response.")
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
    df["ner"] = df["ner"].apply(ast.literal_eval)
    df = df[df["ner"].notnull()]  # Remove any NaNs

    all_entities = [entity.strip() for sublist in df["ner"] for entity in sublist if entity.strip()]
    unique_entities = list(set(all_entities))
    logging.info(f"Found {len(unique_entities)} unique entities.")

    try:
        with open(NORMALIZED_ENTITIES_FILE, "r", encoding="utf-8") as f:
            normalized_entities = json.load(f)
            logging.info(f"Loaded {len(normalized_entities)} previously normalized entities.")
    except (FileNotFoundError, json.JSONDecodeError):
        normalized_entities = {}

    remaining_entities = [e for e in unique_entities if e not in normalized_entities]

    logging.info("Normalizing entities...")
    total_normalized = len(normalized_entities)
    pbar = tqdm(batch_entities(remaining_entities, BATCH_SIZE), desc="Normalizing", total=ceil(len(remaining_entities) / BATCH_SIZE))

    for batch in pbar:
        prompt = build_prompt(batch)
        logging.info(f"Prompt built with {len(batch)} entities.")
        result = query_llm(prompt, client)

        if not result:
            logging.warning("Received empty response from LLM, skipping batch.")
            continue

        normalized_entities.update(result)
        total_normalized += len(result)
        pbar.set_postfix_str(f"{total_normalized}/{len(unique_entities)} normalized")

        # Save progress
        with open(NORMALIZED_ENTITIES_FILE, "w", encoding="utf-8") as f:
            json.dump(normalized_entities, f, ensure_ascii=False, indent=2)
        logging.info(f"Checkpoint saved: {len(normalized_entities)} entities normalized so far.")

        sleep(0.5)

    logging.info("Applying normalization to dataset...")
    df["ner"] = df["ner"].apply(lambda ents: [normalized_entities.get(ent, ent) for ent in ents])
    df.to_csv(NORMALIZED_ENTITIES_DATASET_FILE, index=False)
    logging.info(f"Normalized dataset saved to {NORMALIZED_ENTITIES_DATASET_FILE}")
