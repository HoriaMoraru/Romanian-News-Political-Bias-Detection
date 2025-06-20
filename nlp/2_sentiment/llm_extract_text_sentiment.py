import pandas as pd
from openai import OpenAI
import re
import logging
from tqdm import tqdm
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL = "/models/Llama-3.3-70B-Instruct-bnb-4bit"
TEMPERATURE = 0.0
MAX_TOKENS = 50
MAX_CONTEXT_TOKENS = 8000
INPUT_FILE = "dataset/romanian_political_articles_v2_snorkel.csv"

def build_prompt(text):
    prompt = f"""
    Text: "{text[:MAX_CONTEXT_TOKENS].replace('\n', ' ')}"
        Ești un model care etichetează sentimente în limba română.

        Întrebare: Care este sentimentul general (pozitiv, negativ, neutru)?
        Și menționează un scor de încredere între 0 și 1.

        Format răspuns:
        Sentiment: <pozitiv/negativ/neutru>
        Încredere: <număr între 0.0-1.0>
    """
    return prompt

def query_llama_sentiment(text, client):
    prompt = build_prompt(text)
    try:
        response = client.completions.create(
            model=MODEL,
            prompt=prompt,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        message = response.choices[0].text.strip()

        sentiment_match = re.search(r"Sentiment:\s*(pozitiv|negativ|neutru)", message, re.IGNORECASE)
        confidence_match = re.search(r"Încredere:\s*(0\.\d+|1\.0+)", message)

        sentiment = sentiment_match.group(1).lower() if sentiment_match else "unknown"
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0
        logging.info(f"Processed text: {text[:50]}... Sentiment: {sentiment}, Confidence: {confidence}")

        return sentiment, confidence
    except Exception as e:
        logging.warning(f"Error processing text: {e}")
        return None, None

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
    df = pd.read_csv(INPUT_FILE)

    logging.info("Splitting dataset into 8 parts...")
    total = len(df)
    part_size = total // 8
    start = (split_part - 1) * part_size
    end = start + part_size if split_part < 8 else total
    df = df.iloc[start:end]

    logging.info("Processing rows...")
    tqdm.pandas(desc=f"[Part {split_part}] Rows {start}-{end}")
    results = df["cleantext"].fillna("").apply(lambda text: query_llama_sentiment(text, client))
    df["sentiment_llama"] = results.apply(lambda x: x[0])
    df["sentiment_confidence"] = results.apply(lambda x: x[1])

    output_file = f"dataset/romanian_political_articles_v2_sentiment_part{split_part}.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"Stance extraction completed and saved {len(df)} rows to {output_file}.")
