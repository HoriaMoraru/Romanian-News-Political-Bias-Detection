import pandas as pd
import ollama
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
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjQ4MTJhYmU1LWY0ZmUtNDRlYS1hOGVjLTU5NWRkY2EzNzI0OSJ9.0pC36FM_D-0qb6Wi2TDZb6cakR7tCAkUyJgVwPRtLO8"
MAX_TOKENS = 10
MAX_CONTEXT_TOKENS = 4096

def build_prompt(text: str, entitate: str) -> str:
    prompt = f"""
    # Clasificarea atitudinii față de o entitate politică

    ## Sarcină

    Ai la dispoziție variabila `text` (un fragment extras dintr-un articol de presă politică) în care se menționează o anumită entitate politică, precum și variabila `entitate` (numele entității respective). Sarcina ta este să analizezi acest text și să determini atitudinea exprimată față de entitatea menționată.

    Modelul trebuie să aleagă **una singură** dintre următoarele etichete posibile, conform definițiilor de mai jos:

    ## Etichete posibile

    - **pozitiv**: ton favorabil sau susținere clară față de entitate (inclusiv exprimarea aprecierii, manifestarea încrederii ori evidențierea realizărilor entității în termeni laudativi).
    - **negativ**: ton critic, disprețuitor sau ostil la adresa entității (inclusiv critici directe, exprimarea disprețului, acuzații sau atacuri la adresa entității).
    - **neutru**: relatare obiectivă sau mențiune strict factuală despre entitate (fără exprimarea unei opinii evidente pozitive sau negative).

    Modelul trebuie să rămână imparțial și să nu favorizeze nicio etichetă. Evaluează strict tonul și conținutul textului dat, fără a te lăsa influențat de cunoștințe din afara textului.

    ## Exemple

    1. **Entitate:** "Klaus Iohannis"
        **Text:** "Președintele Klaus Iohannis a fost lăudat pentru inițiativa sa recentă. Mulți au afirmat că el a dat dovadă de o conducere vizionară în gestionarea crizei."
        **Etichetă așteptată:** `pozitiv`

    2. **Entitate:** "PSD"
        **Text:** "PSD a fost criticat dur într-un editorial recent, fiind acuzat de abordări populiste și lipsă de transparență în ultimul an."
        **Etichetă așteptată:** `negativ`

    3. **Entitate:** "Parlamentul României"
        **Text:** "Parlamentul României s-a reunit ieri în ședință comună pentru a discuta modificările propuse la legea bugetului, fără incidente notabile."
        **Etichetă așteptată:** `neutru`

        Acum, analizează textul de mai jos referitor la entitatea **{entitate}** și indică eticheta corectă în funcție de atitudinea exprimată:

        \"\"\"{text}\"\"\"

    Returnează **exclusiv** eticheta adecvată (`pozitiv`, `negativ` sau `neutru`), fără niciun alt comentariu sau explicație.
    """
    return prompt


def query_llm(prompt, client, max_retries=3):

    for _ in range(max_retries):
        try:
            response = client.generate(
                model="deepseek-r1:70b",
                prompt=prompt
            )
            message = response.get('response', '').strip()
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

def analyze_stance(text: str, entities: List[str], client, tokenizer) -> List[dict]:
    results = []
    sentences = sent_tokenize(text)

    for entity in entities:
        relevant_sentences = [s for s in sentences if entity.lower() in s.lower()]
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
    if len(sys.argv) != 2 or sys.argv[1] not in {"1", "2", "3", "4"}:
        print("Usage: python script.py [1|2|3|4]")
        sys.exit(1)

    split_part = int(sys.argv[1])

    logging.info("Loading OpenAI client...")
    client = ollama.Client(
    host='https://chat.readerbench.com/ollama',
    headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
    )

    logging.info("Loading dataset...")
    df = pd.read_csv("../dataset/romanian_political_articles_v2_ner.csv")

    logging.info("Splitting dataset into quarters...")
    total = len(df)
    quarter_size = total // 4
    start = (split_part - 1) * quarter_size
    end = start + quarter_size if split_part < 4 else total
    df = df.iloc[start:end]

    logging.info("Getting tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")

    logging.info("Processing rows...")
    tqdm.pandas(desc=f"[Part {split_part}] Rows {start}–{end}")
    df['stance'] = df.progress_apply(
        lambda row: analyze_stance(row['cleantext'], safe_eval_entities(row['ner']), client, tokenizer), axis=1
    )

    output_file = f"../dataset/romanian_political_articles_v2_sentiment_part{split_part}.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"Stance extraction completed and saved {len(df)} rows to {output_file}.")

