import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newsplease import NewsPlease
import argparse

MODEL_PATH = "./bias-finetuned"
LABELS = ["biased", "unbiased"]

MAX_LEN = 512

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_article_text(url: str) -> str:
    try:
        article = NewsPlease.from_url(url)
        return article.maintext.strip() if article and article.maintext else ""
    except Exception as e:
        logging.warning(f"Error scraping article: {e}")
        return ""


def predict_bias(text: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return LABELS[prediction]


def main():
    parser = argparse.ArgumentParser(description="Detect media bias from an article URL.")
    parser.add_argument("url", help="The URL of the news article.")
    args = parser.parse_args()

    logging.info(f"Fetching article from URL: {args.url}")
    text = extract_article_text(args.url)

    if not text:
        logging.error("No text extracted. Cannot proceed with prediction.")
        return

    logging.info("Predicting bias...")
    result = predict_bias(text)
    logging.info(f"Predicted label: {result}")


if __name__ == "__main__":
    main()
