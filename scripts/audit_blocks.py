#!/usr/bin/env python3
"""Audit parsed blocks for OCR quality issues before training/solving.

Checks per task:
  - empty or missing block files
  - gaps in the printed b01..bNN id sequence (a missed block)
  - suspiciously short blocks (likely truncated transcription)
  - duplicate ids beyond the first (OCR flakiness)

Exit code 1 if any fixable issues found, so it can gate a pipeline.

Usage:
  python scripts/audit_blocks.py --split train            # report only
  python scripts/audit_blocks.py --split train --fix      # re-OCR problem tasks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsem.blocks import normalize_block_id  # noqa: E402
from docsem.data import blocks_path_for, split_tasks  # noqa: E402


def audit_task(path: Path) -> list[str]:
    issues: list[str] = []
    if not path.exists():
        return ["missing"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["unparsable"]
    blocks = data.get("blocks", [])
    if not blocks:
        return ["empty"]
    ids = sorted(int(normalize_block_id(b["id"])[1:]) for b in blocks if normalize_block_id(b.get("id", "")))
    if not ids:
        return ["no-valid-ids"]
    present = set(ids)
    gaps = [f"b{n:02d}" for n in range(1, max(ids)) if n not in present]
    if gaps:
        issues.append(f"id-gaps:{','.join(gaps)}")
    short = [b["id"] for b in blocks if 0 < len(b.get("text", "")) < 40]
    if short:
        issues.append(f"short-blocks:{','.join(short)}")
    dupes = len(ids) - len(present)
    if dupes:
        issues.append(f"duplicate-ids:{dupes}")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--fix", action="store_true", help="delete problem block files so ocr_blocks.py re-OCRs them")
    ap.add_argument("--min-text-len", type=int, default=40)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tasks = split_tasks(args.split, data_dir)
    flagged = 0
    deleted = 0
    for task in tasks:
        tid = task["instance_id"]
        bp = blocks_path_for(tid, args.split, data_dir)
        issues = audit_task(bp)
        if not issues:
            continue
        flagged += 1
        print(f"{tid}: {'; '.join(issues)}")
        if args.fix and bp.exists():
            bp.unlink()
            deleted += 1

    print(f"\n{args.split}: {flagged}/{len(tasks)} tasks flagged"
          + (f", {deleted} files deleted (re-run ocr_blocks.py to refill)" if args.fix else ""))
    if flagged and not args.fix:
        print("Run with --fix to delete flagged files, then re-run ocr_blocks.py.")
        sys.exit(1)


if __name__ == "__main__":
    main()
