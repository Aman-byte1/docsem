"""OCR verification: second-pass checks that catch digit-level transcription errors.

Why: on scanned documents the costliest OCR failures are silent — a "6" read as
"b", a missed block id, a truncated block. The solver then computes confidently
from wrong numbers. This module adds two cheap guards:

1. verify_blocks(): re-reads each page with a verifier prompt that focuses on
   numbers/ids, then MERGES the two readings (union of block ids, longest text
   wins only when both agree on most content — otherwise prefer the verifier's
   reading for numbers).
2. ocr_task_with_retry(): after a normal pass, if the page is missing expected
   ids (a gap in the b01..bNN sequence) or blocks look truncated, re-OCR with an
   explicit hint listing the missing ids.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import OCR_PROMPT, VERIFIER_PROMPT
from .llm import LLMClient
from .normalize import extract_json_array

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_ID_START = re.compile(r"^\s*\bb\s*[oO0]?\s*(\d{1,2})\s*[:.]", re.IGNORECASE)


def _parse_blocks(raw: str) -> list[dict]:
    arr = extract_json_array(raw) or []
    from .blocks import normalize_block_id

    out = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        bid = normalize_block_id(str(item.get("id", "")))
        text = str(item.get("text", "")).strip()
        if bid and text:
            out.append({"id": bid, "text": text})
    return out


def _numbers(text: str) -> set[str]:
    return set(_NUM_RE.findall(text))


def _agree(a: str, b: str) -> float:
    """Rough agreement: Jaccard over number sets + keyword overlap in [0,1]."""
    na, nb = _numbers(a), _numbers(b)
    wa = set(re.findall(r"[a-z]{3,}", a.lower()))
    wb = set(re.findall(r"[a-z]{3,}", b.lower()))
    num_j = len(na & nb) / max(1, len(na | nb))
    word_j = len(wa & wb) / max(1, len(wa | wb))
    return 0.7 * num_j + 0.3 * word_j


def verify_page(
    client: LLMClient,
    page_png: Path,
    first_pass: list[dict],
    max_tokens: int = 2048,
) -> list[dict]:
    """Second-pass verification of one page. Returns the merged block list.

    Merge policy per block id present in either reading:
      - id in both and readings agree (>=0.8): keep the first pass (cheaper trust)
      - id in both and readings disagree: prefer the verifier's text UNLESS the
        first pass has strictly more numbers — the verifier prompt biases toward
        over-editing; disagreement most often means a digit, and we keep the
        reading with the fuller number set, verifier wins ties.
      - id in only one: keep that one (the other pass dropped it).
    """
    if not first_pass:
        return []
    import json as _json

    prompt = VERIFIER_PROMPT.format(bid="page", json=_json.dumps(first_pass, ensure_ascii=False))
    raw = client.chat_with_images(prompt, [page_png], temperature=0.0, max_tokens=max_tokens)
    second = _parse_blocks(raw)
    if not second:
        return first_pass

    first_by_id = {b["id"]: b["text"] for b in first_pass}
    second_by_id = {b["id"]: b["text"] for b in second}
    merged: dict[str, dict] = {}
    for bid in sorted(set(first_by_id) | set(second_by_id)):
        a, b = first_by_id.get(bid), second_by_id.get(bid)
        if a is not None and b is not None:
            if _agree(a, b) >= 0.8:
                text = a
            else:
                na, nb = len(_numbers(a)), len(_numbers(b))
                text = a if na > nb else b
        else:
            text = a if a is not None else b
        merged[bid] = {"id": bid, "text": text}
    return [merged[k] for k in sorted(merged)]


def expected_missing_ids(blocks: list[dict]) -> list[str]:
    """Find gaps in the b01..bNN id sequence (printed ids are dense in this task).

    e.g. blocks b01,b02,b04,b05 -> ['b03']. Only flags interior gaps; the
    document's true last id is unknown, so trailing ids are never "missing".
    """
    ids = sorted(int(b["id"][1:]) for b in blocks)
    if not ids:
        return []
    present = set(ids)
    missing = []
    for n in range(1, max(ids)):
        if n not in present:
            missing.append(f"b{n:02d}")
    return missing


def verify_task(
    client: LLMClient,
    pages: list[Path],
    first_pass_blocks: list[dict],
    *,
    max_tokens: int = 2048,
) -> tuple[list[dict], dict]:
    """Verify all pages of a task; returns (blocks, stats)."""
    per_page: dict[int, list[dict]] = {}
    for b in first_pass_blocks:
        per_page.setdefault(int(b.get("page", 1)), []).append(b)

    stats = {"pages": len(pages), "changed": 0, "added": 0}
    out: list[dict] = []
    for pno, png in enumerate(pages, start=1):
        first = per_page.get(pno, [])
        verified = verify_page(client, png, first, max_tokens=max_tokens)
        old_ids = {b["id"] for b in first}
        new_ids = {b["id"] for b in verified}
        stats["added"] += len(new_ids - old_ids)
        for b in verified:
            prev = next((x["text"] for x in first if x["id"] == b["id"]), None)
            if prev is not None and prev != b["text"]:
                stats["changed"] += 1
            b["page"] = pno
            out.append(b)

    # If ids are still missing after verification, one targeted retry with a hint.
    missing = expected_missing_ids(out)
    if missing and pages:
        hint = (
            "IMPORTANT: a previous transcription MISSED these block ids that are "
            f"printed on the page: {', '.join(missing)}. Transcribe EVERY block, "
            "especially the ones just listed."
        )
        retry_raw = client.chat_with_images(
            OCR_PROMPT + "\n" + hint, pages, temperature=0.0, max_tokens=max_tokens * len(pages)
        )
        retry = _parse_blocks(retry_raw)
        by_id = {b["id"]: b for b in out}
        for b in retry:
            if b["id"] not in by_id or _numbers(b["text"]) - _numbers(by_id[b["id"]]["text"]):
                if b["id"] not in by_id:
                    stats["added"] += 1
                b["page"] = 1  # retry spans all pages; page info becomes unreliable
                by_id[b["id"]] = b
        out = [by_id[k] for k in sorted(by_id)]

    return out, stats
