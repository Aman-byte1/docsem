"""Block parsing and retrieval helpers.

Block ids are printed in the scanned documents as "b01:", "b02:", ... (sometimes OCR'd
as "bo1:", "b 10:" etc.). We normalize any of those to canonical zero-padded ids.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

ID_RE = re.compile(r"\bb\s*[oO0]?\s*(\d{1,2})\s*[:.]", re.IGNORECASE)
BOILERPLATE = re.compile(
    r"^(page\s*\d+|docsem\s*\|?\s*training\s*copy|docsem|training\s*copy|cop|training)$",
    re.IGNORECASE,
)

TOPIC_PATTERNS = [
    re.compile(r"concerning\s+([a-z]+(?:\s+and\s+[a-z]+)*)", re.IGNORECASE),
    re.compile(r"about\s+([a-z]+(?:\s+and\s+[a-z]+)*)", re.IGNORECASE),
    re.compile(r"scenario\s+(?:about|concerning)\s+([a-z]+(?:\s+and\s+[a-z]+)*)", re.IGNORECASE),
]


def normalize_block_id(raw: str) -> str | None:
    """'B10', 'bo3', 'b 10:', 'b10' -> 'b10' (two-digit zero-padded), else None."""
    m = re.search(r"\bb\s*[oO0]?\s*(\d{1,2})\s*[:.]?", raw.strip())
    if not m:
        return None
    return f"b{int(m.group(1)):02d}"


def is_boilerplate_line(text: str) -> bool:
    return bool(BOILERPLATE.match(text.strip()))


def topic_words(query: str) -> list[str]:
    """Extract the paraphrased scenario keywords from a user query, e.g.
    '...concerning percentage and foot...' -> ['percentage', 'foot']."""
    for pat in TOPIC_PATTERNS:
        m = pat.search(query)
        if m:
            words = [w.strip().lower() for w in m.group(1).split("and")]
            return [w for w in words if w]
    return []


def load_blocks(blocks_json: str | Path) -> list[dict]:
    """Load a parsed blocks file. Returns [{'id','text','page'}, ...] sorted by id."""
    with open(blocks_json, encoding="utf-8") as f:
        data = json.load(f)
    blocks = []
    for b in data.get("blocks", []):
        bid = normalize_block_id(str(b.get("id", "")))
        text = (b.get("text") or "").strip()
        if bid and text:
            blocks.append({"id": bid, "text": text, "page": int(b.get("page", 0))})
    # de-duplicate by id (keep the longest text)
    by_id: dict[str, dict] = {}
    for b in blocks:
        if b["id"] not in by_id or len(b["text"]) > len(by_id[b["id"]]["text"]):
            by_id[b["id"]] = b
    return [by_id[k] for k in sorted(by_id)]


def blocks_to_text(blocks: Iterable[dict]) -> str:
    """Format blocks for a prompt."""
    return "\n".join(f"[{b['id']}] {b['text']}" for b in blocks)


# ---------------------------------------------------------------------------
# Fallback lexical scoring (used when no trained selector is available)
# ---------------------------------------------------------------------------

def score_blocks_lexical(query: str, blocks: list[dict]) -> list[tuple[str, float]]:
    """Score blocks by BM25 + scenario-topic overlap. Returns [(id, score)] desc."""
    from rank_bm25 import BM25Okapi

    docs = [b["text"].lower().split() for b in blocks]
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(query.lower().split())

    topics = topic_words(query)
    out = []
    for i, b in enumerate(blocks):
        text_l = b["text"].lower()
        overlap = sum(1 for w in topics if w in text_l)
        question_bonus = 0.5 if re.search(r"\?|\bwhat\b|\bhow many\b|\bhow much\b", text_l) else 0.0
        out.append((b["id"], scores[i] + 2.0 * overlap + question_bonus))
    out.sort(key=lambda x: x[1], reverse=True)
    return out