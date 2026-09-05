#!/usr/bin/env python3
"""Evaluate a predictions JSONL against train labels.

Metrics (per the DocSem protocol):
  - answer accuracy: normalized exact match
  - evidence exact block-set match
  - evidence F1 (diagnostic)

Usage:
  python scripts/evaluate.py --predictions data/predictions/train/predictions.jsonl
  python scripts/evaluate.py --predictions data/predictions/train/predictions.jsonl --errors 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsem.data import load_labels  # noqa: E402
from docsem.metrics import evaluate, report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--errors", type=int, default=0, help="print up to N wrong instances")
    args = ap.parse_args()

    preds = [json.loads(l) for l in open(args.predictions, encoding="utf-8") if l.strip()]
    labels = load_labels(Path(args.data_dir) / "train" / "labels.jsonl")

    known = [p for p in preds if p["instance_id"] in labels]
    print(f"predictions: {len(preds)}  with labels: {len(known)}")
    if not known:
        sys.exit("no labeled predictions to evaluate (are these train predictions?)")

    metrics = evaluate(known, labels)
    print("RESULT:", report(metrics))

    if args.errors:
        from docsem.metrics import answer_em, evidence_exact

        wrong = [p for p in known if not answer_em(p.get("answer"), labels[p["instance_id"]]["answer"])]
        print(f"\nAnswer errors ({len(wrong)}):")
        for p in wrong[: args.errors]:
            g = labels[p["instance_id"]]
            print(f"  {p['instance_id']}: pred={p.get('answer')} gold={g['answer']} "
                  f"pred_ev={p.get('evidence')} gold_ev={g['evidence']}")
        ev_wrong = [p for p in known if answer_em(p.get("answer"), labels[p["instance_id"]]["answer"])
                    and not evidence_exact(p.get("evidence") or [], labels[p["instance_id"]]["evidence"])]
        if ev_wrong:
            print(f"\nEvidence-only errors ({len(ev_wrong)}):")
            for p in ev_wrong[: args.errors]:
                g = labels[p["instance_id"]]
                print(f"  {p['instance_id']}: pred_ev={p.get('evidence')} gold_ev={g['evidence']}")


if __name__ == "__main__":
    main()