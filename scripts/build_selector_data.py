#!/usr/bin/env python3
"""Build (query, block_text, label) training pairs for the evidence selector.

Run AFTER ocr_blocks.py so every train task has a parsed blocks JSON.

Pairs per task:
  - positive : each gold evidence block (usually exactly one)
  - hard negatives : blocks that share >=1 scenario topic word with the gold
                     block but are not gold (the classic "right topic, wrong
                     block" distractors)
  - easy negatives : up to `--negatives` random other blocks

Output: data/selector/{train,dev}.jsonl  (dev = 10% holdout by instance id)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsem.blocks import load_blocks, topic_words  # noqa: E402
from docsem.data import blocks_path_for, load_labels, split_tasks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--negatives", type=int, default=8, help="max negatives per task")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rng = random.Random(args.seed)

    tasks = split_tasks("train", data_dir)
    labels = load_labels(data_dir / "train" / "labels.jsonl")

    rows = []
    skipped = 0
    for task in tasks:
        tid = task["instance_id"]
        lab = labels.get(tid)
        bp = blocks_path_for(tid, "train", data_dir)
        if not lab or not bp.exists():
            skipped += 1
            continue
        blocks = load_blocks(bp)
        if not blocks:
            skipped += 1
            continue
        gold_ids = set(lab["evidence"])
        gold_texts = [b["text"] for b in blocks if b["id"] in gold_ids]
        if not gold_texts:
            skipped += 1
            continue
        non_gold = [b for b in blocks if b["id"] not in gold_ids]
        gold_topics = set(topic_words(task["user_query"])) or set(
            w for t in gold_texts for w in t.lower().split() if len(w) > 3
        )
        gold_page = next((b["page"] for b in blocks if b["id"] in gold_ids), 0)

        def q_like(b: dict) -> bool:
            return bool(re.search(r"\?|\bwhat\b|\bhow many\b|\bhow much\b", b["text"].lower()))

        def has_num(b: dict) -> bool:
            return bool(re.search(r"\d", b["text"]))

        # Hard negatives in priority order: same-topic, question-like, same-page,
        # numeric, then random. Deduped by id, capped at --negatives per task.
        pools = [
            ("topic", lambda b: any(w in b["text"].lower() for w in gold_topics)),
            ("question", q_like),
            ("same_page", lambda b: gold_page and b.get("page") == gold_page),
            ("numeric", has_num),
            ("random", lambda b: True),
        ]
        chosen_ids: set[str] = set()
        negatives: list[dict] = []
        for _name, pred in pools:
            need = args.negatives - len(negatives)
            if need <= 0:
                break
            cand = [b for b in non_gold if b["id"] not in chosen_ids and pred(b)]
            rng.shuffle(cand)
            picked = cand[:need]
            negatives.extend(picked)
            chosen_ids.update(b["id"] for b in picked)

        query = task["user_query"]
        for gt in gold_texts:
            rows.append({"instance_id": tid, "query": query, "text": gt, "label": 1})
        for b in negatives:
            rows.append({"instance_id": tid, "query": query, "text": b["text"], "label": 0})

    rng.shuffle(rows)
    valid_tids = sorted(list({r["instance_id"] for r in rows}))
    dev_ids = set(rng.sample(valid_tids, max(1, int(len(valid_tids) * args.dev_frac))))
    dev = [r for r in rows if r["instance_id"] in dev_ids]
    train = [r for r in rows if r["instance_id"] not in dev_ids]

    out_dir = data_dir / "selector"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("dev", dev)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # persist the exact dev split so train_selector --eval-recall uses the same tasks
    with open(out_dir / "dev_ids.json", "w", encoding="utf-8") as f:
        json.dump(sorted(dev_ids), f)
    pos = sum(1 for r in rows if r["label"] == 1)
    print(f"tasks used: {len(tasks) - skipped}/{len(tasks)} (skipped {skipped})")
    print(f"pairs: train={len(train)} dev={len(dev)}  positives={pos}  negatives={len(rows) - pos}")
    print(f"dev tasks: {len(dev_ids)}")


if __name__ == "__main__":
    main()