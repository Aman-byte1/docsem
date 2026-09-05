#!/usr/bin/env python3
"""Sync heavy pipeline artifacts to/from the private HF dataset repo.

This is how a NEW pod skips all finished work: push the parsed blocks/selector/
predictions once, and on any future pod `pull` restores them instantly.

Repo: amanuelbyte/docsem (private HF dataset)
Dirs synced: data/blocks, data/predictions, data/submission, models/selector

Usage (on the pod or locally):
  export HF_TOKEN=hf_xxx          # or run `huggingface-cli login` once
  python scripts/hf_sync.py push  # upload local artifacts to HF
  python scripts/hf_sync.py pull  # download artifacts from HF (fills gaps, never deletes)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = os.environ.get("DOCSEM_HF_REPO", "amanuelbyte/docsem")
DIRS = [
    "data/blocks",
    "data/predictions",
    "data/submission",
    "models/selector",
]


def push() -> None:
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    for d in DIRS:
        p = Path(d)
        if not p.exists() or not any(p.rglob("*")):
            print(f"skip {d} (missing or empty)")
            continue
        n = sum(1 for f in p.rglob("*") if f.is_file())
        print(f"push {d} ({n} files) -> {REPO}")
        api.upload_folder(
            repo_id=REPO,
            repo_type="dataset",
            folder_path=str(p),
            path_in_repo=d,
        )
    print(f"Pushed to https://huggingface.co/datasets/{REPO} (private)")


def pull() -> None:
    patterns = [f"{d}/**" for d in DIRS]
    snapshot_download(
        REPO,
        repo_type="dataset",
        local_dir=".",
        allow_patterns=patterns,
    )
    for d in DIRS:
        p = Path(d)
        n = sum(1 for f in p.rglob("*") if f.is_file()) if p.exists() else 0
        print(f"{d}: {n} files")
    print("Pull complete.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("direction", choices=["push", "pull"])
    args = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        # huggingface_hub also reads `huggingface-cli login` state; only warn.
        print("NOTE: HF_TOKEN not set — relying on cached login if any.", file=sys.stderr)
    (push if args.direction == "push" else pull)()


if __name__ == "__main__":
    main()
