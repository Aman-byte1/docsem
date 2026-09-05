#!/usr/bin/env python3
"""Train the evidence selector: a DeBERTa cross-encoder scoring (query, block).

This is the supervised Stage-1 "evidence policy" — it learns which block a
paraphrased query points at, from the 908 gold labels. Small data (~7-8k pairs),
so a DeBERTa-v3-small trains in minutes on an A100.

Usage (RunPod):
  python scripts/train_selector.py --data-dir data --output models/selector

After training, the same script evaluates recall@k on the held-out dev tasks by
ranking each dev document's blocks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from docsem.blocks import load_blocks  # noqa: E402
from docsem.data import blocks_path_for, load_labels, split_tasks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output", default="models/selector")
    ap.add_argument("--model-name", default="microsoft/deberta-v3-small")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--eval-recall", action="store_true", help="rank dev blocks after training")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "selector" / "train.jsonl"
    dev_path = data_dir / "selector" / "dev.jsonl"
    if not train_path.exists():
        sys.exit(f"Missing {train_path} — run scripts/build_selector_data.py first.")

    def load(path):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        return Dataset.from_list(rows)

    train_ds, dev_ds = load(train_path), load(dev_path)
    print(f"train pairs: {len(train_ds)}, dev pairs: {len(dev_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def tok(examples):
        return tokenizer(
            examples["query"], examples["text"],
            truncation=True, padding="max_length", max_length=args.max_length,
        )

    train_ds = train_ds.map(tok, batched=True, remove_columns=["instance_id", "query", "text"])
    dev_ds = dev_ds.map(tok, batched=True, remove_columns=["instance_id", "query", "text"])
    train_ds = train_ds.rename_column("label", "labels")
    dev_ds = dev_ds.rename_column("label", "labels")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # transformers >=4.46 renamed evaluation_strategy -> eval_strategy
    import inspect

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_kw = (
        {"eval_strategy": "epoch"}
        if "eval_strategy" in ta_params
        else {"evaluation_strategy": "epoch"}
    )
    args_t = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        **eval_strategy_kw,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    # transformers >=5.0 renamed Trainer(tokenizer=...) -> Trainer(processing_class=...)
    trainer_params = inspect.signature(Trainer.__init__).parameters
    tokenizer_kw = (
        {"processing_class": tokenizer}
        if "processing_class" in trainer_params
        else {"tokenizer": tokenizer}
    )
    trainer = Trainer(
        model=model,
        args=args_t,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        **tokenizer_kw,
    )
    print("Training ...")
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved selector to {out_dir}")

    if args.eval_recall:
        evaluate_recall(out_dir, data_dir)


def evaluate_recall(selector_dir: Path, data_dir: Path, k_values=(1, 3, 5, 8)) -> None:
    """Rank each dev task's blocks and measure recall@k of the gold block."""
    from docsem.selector import EvidenceSelector

    sel = EvidenceSelector(selector_dir)
    tasks = split_tasks("train", data_dir)
    labels = load_labels(data_dir / "train" / "labels.jsonl")
    dev_ids_file = data_dir / "selector" / "dev_ids.json"
    if dev_ids_file.exists():
        import json as _json

        dev_ids = set(_json.loads(dev_ids_file.read_text(encoding="utf-8")))
    else:
        print("WARNING: data/selector/dev_ids.json not found — falling back to first 10% by id")
        dev_ids = set(sorted(t["instance_id"] for t in tasks)[: max(1, int(len(tasks) * 0.1))])
    hits = {k: 0 for k in k_values}
    n_tasks = 0
    for task in tasks:
        tid = task["instance_id"]
        if tid not in dev_ids:
            continue
        bp = blocks_path_for(tid, "train", data_dir)
        if not bp.exists() or tid not in labels:
            continue
        blocks = load_blocks(bp)
        gold = set(labels[tid]["evidence"])
        ranked = sel.rank_blocks(task["user_query"], blocks)
        for k in k_values:
            top = {i for i, _ in ranked[:k]}
            hits[k] += int(bool(gold & top))
        n_tasks += 1
    print(f"selector recall@{k_values} over {n_tasks} dev tasks:")
    for k in k_values:
        print(f"  recall@{k} = {hits[k] / n_tasks:.4f}")


if __name__ == "__main__":
    main()