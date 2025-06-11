import re
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from preprocessing.Preprocessor import Preprocessor

# ───────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────
INPUT_FILE             = "dataset/romanian_political_articles_v2_shuffled.csv"
MODEL_NAME             = "intfloat/multilingual-e5-large-instruct"
MAX_POSITION_EMBEDDINGS = 512
OUTPUT_ART_NPY         = "dataset/nlp/bert_article_embeddings.npy"
OUTPUT_ART_CSV         = "dataset/nlp/bert_article_embeddings.csv"
OUTPUT_WORD_NPY        = "dataset/nlp/bert_word_embeddings.npy"
OUTPUT_WORD_JSON       = "dataset/nlp/bert_word_index.json"
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_article_embeddings_vector(
    cleantext: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device
) -> torch.Tensor:
    tokens = tokenizer.encode(cleantext, add_special_tokens=True)
    chunks: list[list[int]] = [tokens[i:i + MAX_POSITION_EMBEDDINGS]
                                for i in range(0, len(tokens), MAX_POSITION_EMBEDDINGS)]

    chunk_embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            input_ids     = torch.tensor([chunk], device=device)
            attention_mask= torch.ones_like(input_ids, device=device)
            outputs       = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden   = outputs.last_hidden_state

            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed        = torch.sum(last_hidden * mask_expanded, dim=1)
            mask_sum      = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_chunk  = (summed / mask_sum).squeeze(0)
            chunk_embeddings.append(pooled_chunk)

    if len(chunk_embeddings) > 1:
        return torch.stack(chunk_embeddings, dim=0).mean(dim=0)
    return chunk_embeddings[0]


def skip_article(text: str, min_words: int = 30) -> bool:
    if not text or not text.strip():
        return True
    return len(re.findall(r"\w+", text)) < min_words


def get_word_embeddings(
    vocab: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 256
) -> tuple[np.ndarray, dict]:
    model.eval()
    embeddings: list[np.ndarray] = []
    token_to_idx: dict[str, int] = {}
    dim = model.config.hidden_size

    for start in tqdm(range(0, len(vocab), batch_size), desc="Embedding tokens"):
        batch = vocab[start:start+batch_size]
        enc = tokenizer(batch, add_special_tokens=False,
                        return_tensors="pt", padding=True,
                        truncation=True, max_length=tokenizer.model_max_length)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        with torch.no_grad():
            last_hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # mean-pool
            pooled = last_hidden.mean(dim=1).cpu().numpy()

        for i, token in enumerate(batch):
            token_to_idx[token] = start + i
            embeddings.append(pooled[i])

    emb_array = np.stack(embeddings, axis=0)
    return emb_array, token_to_idx


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info(f"Loading tokenizer and model ({MODEL_NAME})…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # --- Article Embeddings ---
    logging.info(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["maintext", "source_domain", "url"])
    preprocessor = Preprocessor()
    tqdm.pandas(desc="Cleaning...")
    df['cleantext'] = df['maintext'].progress_apply(lambda t: preprocessor.process_nlp(t))
    df = df[~df['cleantext'].apply(skip_article)]
    logging.info(f"Filtered dataset to {len(df)} by removing short articles.")

    urls, sources, art_embs = [], [], []
    for url, src, text in tqdm(zip(df['url'], df['source_domain'], df['cleantext']),
                                total=len(df), desc="Article embeddings"):
        if skip_article(text): continue
        try:
            vec = get_article_embeddings_vector(text, tokenizer, model, device)
            arr = vec.cpu().numpy()
            urls.append(url); sources.append(src); art_embs.append(arr)
        except Exception as e:
            logging.error(f"Error on {url}: {e}")

    art_emb_array = np.stack(art_embs, axis=0)
    logging.info(f"Computed article embeddings shape: {art_emb_array.shape}")
    np.save(OUTPUT_ART_NPY, art_emb_array)
    cols = [f"dim_{i}" for i in range(art_emb_array.shape[1])]
    df_out = pd.DataFrame(art_emb_array, columns=cols)
    df_out.insert(0, 'source', sources)
    df_out.insert(0, 'url', urls)
    df_out.to_csv(OUTPUT_ART_CSV, index=False)
    logging.info("Saved article embeddings.")

    # --- Word Embeddings ---
    logging.info("Extracting vocabulary from tokenizer...")
    vocab = list(tokenizer.get_vocab().keys())
    logging.info(f"Vocab size: {len(vocab)} tokens")
    word_emb_array, token_to_idx = get_word_embeddings(vocab, tokenizer, model, device)
    logging.info(f"Computed word embeddings shape: {word_emb_array.shape}")

    logging.info(f"Saving word embeddings to {OUTPUT_WORD_NPY}...")
    np.save(OUTPUT_WORD_NPY, word_emb_array)
    logging.info(f"Saving token index mapping to {OUTPUT_WORD_JSON}...")
    with open(OUTPUT_WORD_JSON, 'w', encoding='utf-8') as f:
        json.dump(token_to_idx, f, ensure_ascii=False, indent=2)

    logging.info("Finished embedding extraction for both articles and tokens.")
