"""Program-of-Thought solver with self-consistency.

For a task we:
1. score candidate blocks (trained selector, else lexical fallback),
2. prompt the LLM N times for {evidence, python_code, answer},
3. execute the python (sandboxed) and trust the executed result over the stated answer,
4. majority-vote the normalized answer; evidence = majority evidence among the
   winning answer group, cleaned to valid block ids.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .blocks import load_blocks, normalize_block_id, score_blocks_lexical, blocks_to_text
from .config import SOLVE_PROMPT, DEFAULT_TOP_K
from .llm import LLMClient
from .normalize import exec_python, extract_json_object, normalize_answer


def _parse_solution(text: str) -> dict[str, Any] | None:
    obj = extract_json_object(text)
    if not obj:
        return None
    code = obj.get("python_code") or ""
    if isinstance(code, str) and code.strip().startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n?", "", code).rstrip("`").strip()
    ans = obj.get("answer")
    ev = obj.get("evidence") or []
    if isinstance(ans, (int, float)):
        ans = normalize_answer(str(ans))
    elif isinstance(ans, str):
        ans = normalize_answer(ans)
    else:
        ans = None
    if isinstance(ev, str):
        ev = [ev]
    ev = [e for e in ev if normalize_block_id(str(e))]
    return {"evidence": ev, "python_code": code, "answer": ans, "raw": text[:500]}


def _candidates(query: str, blocks: list[dict], selector, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    if selector is not None:
        ranked = selector.rank_blocks(query, blocks, top_k=top_k)
    else:
        ranked = score_blocks_lexical(query, blocks)[:top_k]
    by_id = {b["id"]: b for b in blocks}
    return [by_id[i] for i, _ in ranked if i in by_id]


def solve_task(
    client: LLMClient,
    llm_model: str,
    query: str,
    blocks: list[dict],
    *,
    selector=None,
    top_k: int = DEFAULT_TOP_K,
    samples: int = 10,
    temperature: float = 0.6,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Returns {answer, evidence, meta:{...}}."""
    candidates = _candidates(query, blocks, selector, top_k)
    if not candidates:
        # No usable blocks: last-resort evidence
        return {
            "answer": None,
            "evidence": [blocks[0]["id"]] if blocks else ["b01"],
            "meta": {"error": "no candidate blocks"},
        }

    prompt = SOLVE_PROMPT.format(query=query, blocks=blocks_to_text(candidates))
    texts = client.chat(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n=samples,
        system="You return only valid JSON. Be precise with arithmetic.",
    )

    parsed = [_parse_solution(t) for t in texts]
    valid = [p for p in parsed if p and (p["answer"] or p["python_code"])]

    # Execute python_code where present; the executed result is ground truth for the answer.
    executed = []
    for p in valid:
        code = p.get("python_code") or ""
        exec_ans = exec_python(code) if code.strip() else None
        p["exec_answer"] = exec_ans
        if exec_ans is not None:
            p["answer"] = exec_ans
        executed.append(p)

    all_ids = {b["id"] for b in blocks}
    cleaned = []
    for p in executed:
        ev = [i for i in p.get("evidence", []) if normalize_block_id(i) and normalize_block_id(i) in all_ids]
        p["evidence"] = ev
        cleaned.append(p)

    # ---- self-consistency: majority vote on normalized answer -----------------
    groups: dict[str, list[dict]] = {}
    for p in cleaned:
        if p.get("answer"):
            groups.setdefault(p["answer"], []).append(p)

    chosen: dict[str, Any] | None = None
    if groups:
        best = max(groups.values(), key=lambda g: (len(g), sum(1 for p in g if p.get("exec_answer"))))
        # within the winning answer group, pick the most common non-empty evidence
        ev_counts: dict[tuple, int] = {}
        for p in best:
            key = tuple(sorted(p.get("evidence", [])))
            if key:
                ev_counts[key] = ev_counts.get(key, 0) + 1
        if ev_counts:
            ev = list(max(ev_counts.items(), key=lambda kv: kv[1])[0])
        elif best and best[0].get("evidence"):
            ev = best[0]["evidence"]
        else:
            ev = []
        chosen = {"answer": best[0]["answer"], "evidence": ev,
                  "meta": {"votes": len(best), "samples": len(cleaned)}}
    elif cleaned:
        # no numeric agreement: prefer a sample with executed code, else first
        picked = next((p for p in cleaned if p.get("exec_answer")), cleaned[0])
        chosen = {"answer": picked.get("answer"), "evidence": picked.get("evidence", []),
                  "meta": {"votes": 1, "samples": len(cleaned), "fallback": "no-majority"}}
    else:
        chosen = {"answer": None, "evidence": [candidates[0]["id"]],
                  "meta": {"votes": 0, "samples": 0, "error": "no parseable samples"}}

    # Ensure evidence is non-empty and valid
    if not chosen["evidence"]:
        chosen["evidence"] = [candidates[0]["id"]]
    chosen["evidence"] = [i for i in chosen["evidence"] if i in all_ids] or [candidates[0]["id"]]
    return chosen


def load_blocks_cached(blocks_json_path) -> list[dict]:
    return load_blocks(blocks_json_path)


def dump_meta(task_id: str, chosen: dict[str, Any]) -> dict[str, Any]:
    return {"instance_id": task_id, **chosen, "meta": json.dumps(chosen.get("meta", {}))}