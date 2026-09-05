#!/usr/bin/env python3
"""Run inference on a split and write predictions JSONL.

Flow per task:
  1. load parsed blocks (data/blocks/{split}/{task_id}.json — run ocr_blocks.py first)
  2. rank candidate blocks with the trained selector (or lexical fallback)
  3. ask the LLM N times for {evidence, python_code, answer}
  4. execute python (sandboxed), majority-vote answers, clean evidence ids
  5. write {instance_id, answer, evidence} to the predictions file

Examples:
  # RunPod (vLLM serving the solver model):
  python scripts/solve.py --split train --llm-url http://localhost:8001/v1 \
      --llm-model Qwen/Qwen2.5-7B-Instruct --selector models/selector \
      --samples 10 --limit 50

  # Local (LM Studio):
  python scripts/solve.py --split train --llm-url http://localhost:1234/v1 \
      --llm-model <model-name> --selector models/selector --limit 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsem.blocks import load_blocks  # noqa: E402
from docsem.data import blocks_path_for, split_tasks  # noqa: E402
from docsem.llm import LLMClient  # noqa: E402
from docsem.selector import load_selector  # noqa: E402
from docsem.solver import solve_task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=0, help="only first N tasks (0 = all)")
    ap.add_argument("--out", default=None, help="predictions JSONL path (default data/predictions/{split}/predictions.jsonl)")
    ap.add_argument("--llm-url", required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:8001/v1")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--selector", default="models/selector", help="trained selector dir (auto-skip if absent)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--samples", type=int, default=10, help="self-consistency samples per task")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="skip instances already in --out")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tasks = split_tasks(args.split, data_dir, args.limit)
    out_path = Path(args.out) if args.out else data_dir / "predictions" / args.split / "predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = LLMClient(base_url=args.llm_url, model=args.llm_model)
    selector = load_selector(args.selector)
    if selector is None:
        print(f"WARNING: no trained selector at {args.selector} — using lexical fallback.")
    else:
        print(f"Using trained selector from {args.selector}")

    done_ids = set()
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done_ids = {json.loads(l)["instance_id"] for l in f if l.strip()}
        print(f"Resuming: {len(done_ids)} already done")

    todo = [t for t in tasks if t["instance_id"] not in done_ids]
    print(f"Solving {len(todo)}/{len(tasks)} tasks -> {out_path}")

    def work(task):
        tid = task["instance_id"]
        bp = blocks_path_for(tid, args.split, data_dir)
        t0 = time.time()
        if not bp.exists():
            return {"instance_id": tid, "answer": None, "evidence": [], "meta": {"error": "no blocks file"}}
        blocks = load_blocks(bp)
        if not blocks:
            return {"instance_id": tid, "answer": None, "evidence": [], "meta": {"error": "empty blocks"}}
        result = solve_task(
            client,
            args.llm_model,
            task["user_query"],
            blocks,
            selector=selector,
            top_k=args.top_k,
            samples=args.samples,
            temperature=args.temperature,
        )
        result["instance_id"] = tid
        result["meta"]["time_s"] = round(time.time() - t0, 2)
        return result

    written = 0
    lock = __import__("threading").Lock()
    meta_path = out_path.with_name("meta.jsonl")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, t) for t in todo]
        for fut in as_completed(futs):
            row = fut.result()
            line = {"instance_id": row["instance_id"], "answer": row.get("answer"), "evidence": row.get("evidence") or []}
            with lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                with open(meta_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"instance_id": row["instance_id"], "meta": row.get("meta", {})}, ensure_ascii=False) + "\n")
            written += 1
            if written % 25 == 0 or written == len(todo):
                print(f"  [{written}/{len(todo)}] {line['instance_id']} answer={line['answer']} evidence={line['evidence']}")

    print(f"Done. Predictions: {out_path}")


if __name__ == "__main__":
    main()