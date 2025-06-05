import re
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from preprocessing.Preprocessor import Preprocessor

# ───────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────
INPUT_FILE = "dataset/romanian_political_articles_v2_shuffled.csv"
MODEL_NAME = "dumitrescustefan/bert-base-romanian-cased-v1"
MAX_POSITION_EMBEDDINGS = 512
OUTPUT_EMB_NPY = "dataset/nlp/bert_article_embeddings.npy"
OUTPUT_EMB_CSV = "dataset/nlp/bert_article_embeddings.csv"
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_article_embeddings_vector(
    cleantext: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device) -> torch.Tensor:
    """
    Given a pre-cleaned text (cleantext), returns a single 768-dim (Model's hidden size)
    embedding (torch.Tensor) by:
      - Tokenizing (with special tokens)
      - Splitting into chunks of at most (512) tokens
      - For each chunk: feed through BERT, extract CLS token
      - Averaging all CLS embeddings into one 768-dim vector
    """
    tokens = tokenizer.encode(cleantext, add_special_tokens=True)
    max_len = MAX_POSITION_EMBEDDINGS

    chunks : list[list[int]]= [tokens[i:i+max_len] for i in range(0, len(tokens), max_len)]

    chunk_embeddings: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            input = torch.tensor([chunk], dtype = torch.long, device=device)               # shape [1, seq_len]
            attention_mask = torch.ones_like(input, device=device)                         # no padding

            outputs = model(input, attention_mask=attention_mask)

            # Use CLS token's embedding (index 0) as chunk representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]                             # shape [1, 768]
            chunk_embeddings.append(cls_embedding[0])                                      # append 1x768 vector

    article_vector = torch.stack(chunk_embeddings).mean(dim=0)                             # shape [768]
    return article_vector

def skip_article(text:str, min_words:int = 30) -> bool:
    """
    Skip articles that are too short or contain only whitespace.
    """
    if not text or not text.strip():
        return True
    words = re.findall(r'\w+', text)
    return len(words) < min_words

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device: {device}")

    logging.info(f"Loading tokenizer and model ({MODEL_NAME})…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    logging.info(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    logging.info("Dropping rows with missing text or source...")
    df = df.dropna(subset=["maintext", "source_domain", "url"])

    logging.info("Loading preprocessor...")
    preprocessor = Preprocessor()

    tqdm.pandas(desc="Cleaning...")
    df['cleantext'] = df['maintext'].progress_apply(lambda t: preprocessor.process_nlp(t))

    df = df[~df['cleantext'].apply(skip_article)]
    logging.info(f"Filtered dataset to {len(df)} by removing short articles.")

    urls: list[str] = []
    sources: list[str] = []
    embeddings: list[np.ndarray] = []

    articles_len = len(df)
    logging.info(f"Computing {MODEL_NAME} embeddings for {articles_len} articles...")

    for url, source, cleantext in tqdm(
        zip(df["url"], df["source_domain"], df["cleantext"]),
        total=articles_len,
        desc="Getting embeddings"
    ):
        if skip_article(cleantext):
            continue
        try:
            article_vector = get_article_embeddings_vector(cleantext, tokenizer, model, device)
            article_vector_np = article_vector.cpu().numpy()  # Convert to numpy array

            urls.append(url)
            sources.append(source)
            embeddings.append(article_vector_np)

        except Exception as e:
            logging.error(f"Error processing article {url}: {e}")

    if not embeddings:
        logging.error("No embeddings were computed. Check your data/skip logic.Exiting...")
        exit(1)

    embeddings_np = np.stack(embeddings, axis = 0)  # Shape: [num_articles, 768]
    logging.info(f"Computed embeddings for {embeddings_np.shape[0]} articles. Shape: {embeddings_np.shape}")

    logging.info(f"Saving embeddings matrix to {OUTPUT_EMB_NPY}...")
    np.save(OUTPUT_EMB_NPY, embeddings_np)

    col_names = [f"dim_{i}" for i in range(embeddings_np.shape[1])]
    df_emb = pd.DataFrame(embeddings_np, columns=col_names)
    df_emb.insert(0, "source", sources)
    df_emb.insert(0, "url", urls)


    logging.info(f"Saving embeddings as csv to {OUTPUT_EMB_CSV}...")
    df_emb.to_csv(OUTPUT_EMB_CSV, index=False)

    logging.info("Finished. You now have:\n"
                f"  • {OUTPUT_EMB_NPY}  → raw NumPy array of shape {embeddings_np.shape}\n"
                f"  • {OUTPUT_EMB_CSV}  → CSV with columns: url, source, dim_0…dim_767")
