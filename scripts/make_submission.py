#!/usr/bin/env python3
"""Produce and validate the official validation submission.

Reads val predictions (or runs solve.py via --run), checks every validation
instance is present exactly once, evidence is non-empty and uses valid block
ids, and writes data/submission/val/submission.jsonl.

Usage:
  python scripts/make_submission.py --predictions data/predictions/val/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsem.blocks import normalize_block_id  # noqa: E402
from docsem.data import split_tasks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="val predictions JSONL (run solve.py --split val first)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    val_tasks = split_tasks("val", data_dir)
    expected_ids = [t["instance_id"] for t in val_tasks]

    preds = [json.loads(l) for l in open(args.predictions, encoding="utf-8") if l.strip()]
    pred_by_id = {p["instance_id"]: p for p in preds}

    missing = [i for i in expected_ids if i not in pred_by_id]
    extra = [i for i in pred_by_id if i not in set(expected_ids)]
    errors: list[str] = []
    for i in expected_ids:
        p = pred_by_id.get(i)
        if not p:
            continue
        if not p.get("answer"):
            errors.append(f"{i}: empty answer")
        ev = p.get("evidence") or []
        if not ev:
            errors.append(f"{i}: empty evidence")
        for e in ev:
            if normalize_block_id(str(e)) is None:
                errors.append(f"{i}: invalid evidence id {e!r}")

    print(f"val instances: {len(expected_ids)}  predictions: {len(preds)}")
    print(f"missing: {len(missing)}  extra: {len(extra)}  schema errors: {len(errors)}")
    for m in missing[:10]:
        print("  MISSING", m)
    for e in errors[:10]:
        print("  ERROR", e)

    if missing or extra or errors:
        sys.exit("Submission invalid — fix before uploading.")

    out = Path(args.out) if args.out else data_dir / "submission" / "val" / "submission.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i in expected_ids:
            p = pred_by_id[i]
            f.write(json.dumps({"instance_id": i, "answer": str(p["answer"]), "evidence": p["evidence"]}, ensure_ascii=False) + "\n")
    print(f"OK: wrote {out} ({len(expected_ids)} lines). Upload to the submission portal.")


if __name__ == "__main__":
    main()