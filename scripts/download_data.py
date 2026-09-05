#!/usr/bin/env python3
"""Download the DocSem shared-task dataset (train + val + examples) into data/.

Uses huggingface_hub snapshot_download (parallel, resumable). Mirrors the
oracle-samples/gsm-sem participant release. ~1.3 GB total.
"""
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

REPO_ID = "amitbcp/docinsights-2026-shared-task-data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="destination directory")
    ap.add_argument("--repo-id", default=REPO_ID)
    args = ap.parse_args()

    print(f"Downloading {args.repo_id} -> {args.data_dir} ...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.data_dir,
        allow_patterns=[
            "train/*",
            "val/*",
            "examples/*",
            "INSTRUCTIONS.md",
            "README.md",
            "LICENSE.txt",
        ],
    )
    print("Done. Check data/train/tasks.jsonl and data/val/tasks.jsonl exist.")


if __name__ == "__main__":
    main()