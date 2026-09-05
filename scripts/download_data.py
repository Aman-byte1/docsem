#!/usr/bin/env python3
"""Download the DocSem shared-task dataset (train + val + examples) into data/.

Uses huggingface_hub snapshot_download (parallel, resumable). Mirrors the
oracle-samples/gsm-sem participant release. ~1.3 GB total.
"""
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

REPO_ID = "amitbcp/docinsights-2026-shared-task-data"


def download_with_retry(repo_id: str, data_dir: str, max_workers: int, max_retries: int = 4, backoff: int = 45) -> None:
    """snapshot_download is resumable (skips already-downloaded files); retry
    transient HF 429 rate limits, which are IP-based and temporary."""
    import time

    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=data_dir,
                allow_patterns=[
                    "train/*",
                    "val/*",
                    "examples/*",
                    "INSTRUCTIONS.md",
                    "README.md",
                    "LICENSE.txt",
                ],
                max_workers=max_workers,
            )
            return
        except Exception as e:  # noqa: BLE001 - surface after retries
            print(f"  attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}")
            if attempt < max_retries:
                print(f"  waiting {backoff}s before retrying (resumes where it left off)...")
                time.sleep(backoff)
    raise RuntimeError("dataset download failed after retries (rate-limited?)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="destination directory")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--max-workers", type=int, default=4, help="parallel downloads (lower = gentler on HF rate limits)")
    args = ap.parse_args()

    print(f"Downloading {args.repo_id} -> {args.data_dir} (max_workers={args.max_workers}) ...")
    download_with_retry(args.repo_id, args.data_dir, args.max_workers)
    print("Done. Check data/train/tasks.jsonl and data/val/tasks.jsonl exist.")


if __name__ == "__main__":
    main()