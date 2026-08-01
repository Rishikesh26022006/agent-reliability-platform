import sys
import json
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "train.jsonl"
VAL_FILE = PROJECT_ROOT / "data" / "val.jsonl"
MODEL_OUT = PROJECT_ROOT / "failure_predictor" / "model_checkpoint"

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import Counter

BASE_MODEL = "distilbert-base-uncased"  # small, fast, good baseline for this data size


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_hf_dataset(rows):
    return Dataset.from_dict({
        "text": [r["text"] for r in rows],
        "label": [int(r["label_is_failure"]) for r in rows],
    })


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class WeightedTrainer(Trainer):
    """Trainer subclass that applies inverse-frequency class weights to the loss.
    This compensates for the class imbalance (failures are the minority class
    but the most important class for the predictor to detect)."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    print("Loading data...")
    train_rows = load_jsonl(TRAIN_FILE)
    val_rows = load_jsonl(VAL_FILE)
    print(f"Train: {len(train_rows)} rows, Val: {len(val_rows)} rows")

    # Show class distribution
    train_labels = [int(r["label_is_failure"]) for r in train_rows]
    val_labels = [int(r["label_is_failure"]) for r in val_rows]
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    print(f"Train class distribution: not_failure={train_counts[0]}, failure={train_counts[1]} ({train_counts[1]/len(train_labels)*100:.1f}%)")
    print(f"Val class distribution:   not_failure={val_counts[0]}, failure={val_counts[1]} ({val_counts[1]/len(val_labels)*100:.1f}%)")

    # Compute inverse-frequency class weights for balanced training
    total = len(train_labels)
    n_classes = 2
    class_weights = torch.tensor([
        total / (n_classes * train_counts[0]),  # weight for class 0 (not failure)
        total / (n_classes * train_counts[1]),  # weight for class 1 (failure)
    ], dtype=torch.float32)
    print(f"Class weights: not_failure={class_weights[0]:.3f}, failure={class_weights[1]:.3f}")

    train_ds = to_hf_dataset(train_rows)
    val_ds = to_hf_dataset(val_rows)

    print(f"Loading tokenizer and model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        num_train_epochs=8,  # more epochs since dataset is small
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,  # warm up over first 10% of steps to stabilize early training
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=5,
        report_to="none",  # skip W&B for now to keep this simple; can add later
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("\nFinal evaluation on validation set:")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"\nSaving model to {MODEL_OUT}")
    trainer.save_model(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))
    print("Done.")