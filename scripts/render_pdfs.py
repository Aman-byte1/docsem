#!/usr/bin/env python3
"""Render every PDF page to a PNG (data/pages/{split}/{task_id}/pNN.png).

The PDFs are full-page scans, so this is the first step of the pipeline for
both the VLM OCR (RunPod) and the rapidocr fallback (local CPU).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsem.data import pdf_path_for, split_tasks  # noqa: E402
from docsem.pdf import render_pdf_pages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0, help="only first N tasks (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tasks = split_tasks(args.split, data_dir, args.limit)
    print(f"Rendering {len(tasks)} tasks @ {args.dpi}dpi ...")

    def work(task):
        tid = task["instance_id"]
        pdf = pdf_path_for(task, data_dir, args.split)
        if not pdf.exists():
            return tid, -1
        out_dir = data_dir / "pages" / args.split / tid
        paths = render_pdf_pages(pdf, out_dir, dpi=args.dpi)
        return tid, len(paths)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            tid, npages = fut.result()
            done += 1
            if npages < 0:
                print(f"  [{done}/{len(tasks)}] SKIP {tid}: pdf not found")
            elif done % 100 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {tid}: {npages} pages")
    print("Done.")


if __name__ == "__main__":
    main()