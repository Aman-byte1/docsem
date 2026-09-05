#!/usr/bin/env python3
"""OCR rendered pages into structured blocks with official bNN: ids.

Two engines:
  --engine vllm     : Qwen2.5-VL served by vLLM (RunPod A100). Best quality.
  --engine rapidocr : CPU-only fallback (local dev). Reconstructs blocks from
                      OCR lines by finding the printed "bNN:" markers.

Output per task: data/blocks/{split}/{task_id}.json
  {"pages": N, "blocks": [{"id": "b01", "text": "...", "page": 1}, ...]}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsem.blocks import is_boilerplate_line, load_blocks, normalize_block_id  # noqa: E402
from docsem.config import OCR_PROMPT  # noqa: E402
from docsem.data import blocks_path_for, pdf_path_for, split_tasks  # noqa: E402
from docsem.llm import LLMClient  # noqa: E402
from docsem.normalize import extract_json_array  # noqa: E402
from docsem.pdf import render_page_image  # noqa: E402


# ---------------------------------------------------------------------------
# VLM engine
# ---------------------------------------------------------------------------

def ocr_page_vlm(client: LLMClient, page_png: Path) -> list[dict]:
    out = client.chat_with_images(OCR_PROMPT, [page_png], temperature=0.0, max_tokens=2048)
    arr = extract_json_array(out)
    if not arr:
        return []
    parsed = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        bid = normalize_block_id(str(item.get("id", "")))
        text = str(item.get("text", "")).strip()
        if bid and text:
            parsed.append({"id": bid, "text": text})
    return parsed


def ocr_task_vlm(client: LLMClient, task: dict, data_dir: Path, split: str) -> tuple[list[dict], int]:
    pages = sorted((data_dir / "pages" / split / task["instance_id"]).glob("p*.png"))
    blocks: list[dict] = []
    for pno, png in enumerate(pages, start=1):
        for b in ocr_page_vlm(client, png):
            b["page"] = pno
            blocks.append(b)
    return blocks, len(pages)


# ---------------------------------------------------------------------------
# rapidocr engine (CPU fallback)
# ---------------------------------------------------------------------------

_ID_START = re.compile(r"^\s*\bb\s*[oO0]?\s*(\d{1,2})\s*[:.]", re.IGNORECASE)


def ocr_task_rapidocr(task: dict, data_dir: Path, split: str, dpi: int) -> tuple[list[dict], int]:
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    pdf = pdf_path_for(task, data_dir, split)
    from docsem.pdf import page_count

    npages = page_count(pdf)
    lines_per_page: list[list[str]] = []
    for pno in range(npages):
        img = render_page_image(pdf, pno, dpi=dpi)
        result, _ = ocr(str(img))
        lines = [t for _, t, _ in (result or [])]
        lines_per_page.append(lines)

    blocks: list[dict] = []
    for pno, lines in enumerate(lines_per_page, start=1):
        cur_id: str | None = None
        cur_text: list[str] = []
        for raw in lines:
            text = raw.strip()
            if not text:
                continue
            m = _ID_START.match(text)
            if m:
                if cur_id:
                    blocks.append({"id": cur_id, "text": " ".join(cur_text)})
                cur_id = f"b{int(m.group(1)):02d}"
                cur_text = [text[m.end():].strip()]
            elif cur_id:
                if not is_boilerplate_line(text):
                    cur_text.append(text)
        if cur_id:
            blocks.append({"id": cur_id, "text": " ".join(cur_text)})
    for b in blocks:
        b["page"] = 1  # page info is approximate here; keep 1-based block order
    return blocks, npages


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--engine", choices=["vllm", "rapidocr"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    # vllm engine
    ap.add_argument("--llm-url", default=None, help="e.g. http://localhost:8000/v1")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    # rapidocr engine
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tasks = split_tasks(args.split, data_dir, args.limit)
    out_dir = data_dir / "blocks" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if args.engine == "vllm":
        if not args.llm_url:
            sys.exit("--engine vllm requires --llm-url (the vLLM OpenAI endpoint)")
        client = LLMClient(base_url=args.llm_url, model=args.llm_model)
        print(f"VLM OCR with {args.llm_model} via {args.llm_url}")

    todo = [t for t in tasks if not blocks_path_for(t["instance_id"], args.split, data_dir).exists()]
    print(f"OCR {len(todo)}/{len(tasks)} tasks (skipping existing) ...")

    def work(task):
        tid = task["instance_id"]
        t0 = time.time()
        pdf = pdf_path_for(task, data_dir, args.split)
        if not pdf.exists():
            return tid, -1, 0.0
        if client is not None:
            blocks, npages = ocr_task_vlm(client, task, data_dir, args.split)
        else:
            blocks, npages = ocr_task_rapidocr(task, data_dir, args.split, args.dpi)
        # dedupe by id, keep longest text
        by_id: dict[str, dict] = {}
        for b in blocks:
            if b["id"] not in by_id or len(b["text"]) > len(by_id[b["id"]]["text"]):
                by_id[b["id"]] = b
        final = [by_id[k] for k in sorted(by_id)]
        with open(blocks_path_for(tid, args.split, data_dir), "w", encoding="utf-8") as f:
            json.dump({"pages": npages, "blocks": final}, f, ensure_ascii=False)
        return tid, len(final), time.time() - t0

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, t) for t in todo]
        for fut in as_completed(futs):
            tid, nblocks, dt = fut.result()
            done += 1
            if nblocks < 0:
                print(f"  [{done}/{len(todo)}] SKIP {tid}: pdf not found")
            elif done % 25 == 0 or done == len(todo):
                print(f"  [{done}/{len(todo)}] {tid}: {nblocks} blocks in {dt:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()