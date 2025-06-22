from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
import wandb
import os
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv("/app/.env")
wandb.login(key=os.getenv("WANDB_API_KEY"))

GOLD_LABELS = "dataset/manual_labels.csv"
WEAK_LABELS = "dataset/snorkel/article_labels.csv"
MODEL_OUTPUT_DIR = "./xlmr-finetuned"

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 512
NUM_LABELS = 2

logging.info(f"Loading tokenizer from {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    logging.info(f"Predictions: {preds}, Labels: {labels}")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds)
    }

def tokenize(example):
    result = tokenizer(
        example["cleantext"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        stride=256,
        return_overflowing_tokens=True,
    )

    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in example.items():
        result[key] = [values[i] for i in sample_map]

    return result

def main():
    logging.info("Loading gold and weak datasets...")
    gold_df = pd.read_csv(GOLD_LABELS)
    weak_df = pd.read_csv(WEAK_LABELS)

    logging.info(f"Gold samples: {len(gold_df)}, Weak samples before deduplication: {len(weak_df)}")

    gold_df = gold_df.rename(columns={"label": "label_text"})
    weak_df = weak_df.rename(columns={"snorkel_label": "label_text"})

    gold_urls = set(gold_df["url"])
    weak_df = weak_df[~weak_df["url"].isin(gold_urls)]

    logging.info(f"Weak samples after removing duplicates with gold: {len(weak_df)}")

    # combined_df = pd.concat([gold_df, weak_df], ignore_index=True)
    combined_df = weak_df
    # combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)


    le = LabelEncoder()
    combined_df["label"] = le.fit_transform(combined_df["label_text"])
    logging.info(f"Label distribution: {np.bincount(combined_df['label'])}")

    num_classes = len(np.unique(combined_df["label"]))
    assert num_classes == NUM_LABELS, f"Expected {NUM_LABELS} classes, but found {num_classes}"

    logging.info("Splitting into train and eval sets...")
    train_df, eval_df = train_test_split(
        combined_df,
        test_size=0.1,
        random_state=42,
        stratify=combined_df["label"]
    )

    logging.info(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}")

    train_ds = Dataset.from_pandas(train_df[["cleantext", "label"]])
    eval_ds = Dataset.from_pandas(eval_df[["cleantext", "label"]])

    logging.info("Tokenizing datasets...")
    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    logging.info(train_ds[0])
    logging.info(eval_ds[0])

    logging.info(f"Loading model {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    args = TrainingArguments(
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        num_train_epochs=6,
        label_names=["labels"],
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        logging_strategy="steps",
        report_to="wandb",
        run_name="political-media-bias-detection",
        output_dir=MODEL_OUTPUT_DIR
    )

    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    logging.info("Starting training...")
    trainer.train()

    metrics = trainer.evaluate()
    logging.info(f"Final evaluation metrics: {metrics}")

    logging.info("Generating classification report...")

    raw_preds = trainer.predict(eval_ds)
    preds = np.argmax(raw_preds.predictions, axis=1)
    labels = raw_preds.label_ids

    report = classification_report(labels, preds, target_names=le.classes_, digits=4)
    logging.info(f"\n{report}")

    logging.info("Saving final model and tokenizer...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    logging.info("Training complete.")

if __name__ == "__main__":
    main()
