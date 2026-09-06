#!/usr/bin/env python3
"""Improve predictions in two phases:

Phase 1 — Evidence: Send ALL blocks to the LLM (not just selector top-8).
  Every other team gets 100% evidence; our selector caps at 88.9% because
  it sometimes excludes the right block from the top-8.

Phase 2 — Answer: Solve with a clean, math-focused prompt on JUST the
  evidence block. Two-pass: first identify the question, then solve it.
  Ensemble with old predictions (keep old answer when models disagree).
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsem.blocks import load_blocks, blocks_to_text
from docsem.data import split_tasks
from docsem.llm import LLMClient
from docsem.normalize import normalize_answer, exec_python, extract_json_object

# ---- Prompts ----

EVIDENCE_PROMPT = """\
Which ONE block below contains a complete, solvable quantitative question \
matching the user's query topics?

USER QUERY: {query}

BLOCKS:
{blocks}

The evidence block states a self-contained math problem with all the numbers \
needed to compute a numerical answer. Distractor blocks mention the same \
topics but do NOT contain a solvable question.
Return ONLY: {{"evidence": "bNN"}}"""

MATH_PROMPT = """\
Solve this math problem step by step. The answer is always an integer.

PROBLEM:
{problem}

Steps:
1. Identify every number given in the problem.
2. Determine exactly what is being asked (percentage? count? difference? total?).
3. Write the arithmetic.
4. Compute the final integer answer.

Return ONLY a JSON object:
{{"python_code": "# step-by-step computation\\nresult = <expression>", "answer": <integer>}}"""


def find_evidence(client, query, blocks, n=5):
    """Majority-vote the evidence block from ALL blocks."""
    all_ids = {b["id"] for b in blocks}
    prompt = EVIDENCE_PROMPT.format(query=query, blocks=blocks_to_text(blocks))
    responses = client.chat(prompt, n=n, temperature=0.3, max_tokens=100)
    votes = Counter()
    for r in responses:
        m = re.search(r'"evidence"\s*:\s*"(b\d+)"', r)
        if m and m.group(1) in all_ids:
            votes[m.group(1)] += 1
    if votes:
        return votes.most_common(1)[0][0], votes
    return None, votes


def solve_question(client, block_text, n=10):
    """Solve a math problem with self-consistency, using both CoT and code exec."""
    prompt = MATH_PROMPT.format(problem=block_text)
    responses = client.chat(prompt, n=n, temperature=0.6, max_tokens=512)

    answer_votes = Counter()
    for r in responses:
        obj = extract_json_object(r)
        if not obj:
            continue
        code = obj.get("python_code", "")
        stated = obj.get("answer")

        # Try code execution first
        ans = None
        if code and code.strip():
            ans = exec_python(code)
        # Fall back to stated answer
        if ans is None and stated is not None:
            ans = normalize_answer(str(stated))
        if ans:
            answer_votes[ans] += 1

    if answer_votes:
        return answer_votes.most_common(1)[0][0], answer_votes
    return None, answer_votes


def process_task(client, task, blocks, old_pred):
    """Process one task: fix evidence + try to improve answer."""
    tid = task["instance_id"]
    query = task["user_query"]

    # Phase 1: find evidence from ALL blocks
    new_ev, ev_votes = find_evidence(client, query, blocks)

    # Phase 2: solve using the evidence block
    evidence_id = new_ev or old_pred["evidence"][0]
    ev_block = next((b for b in blocks if b["id"] == evidence_id), None)
    if ev_block:
        new_ans, ans_votes = solve_question(client, ev_block["text"])
    else:
        new_ans, ans_votes = None, Counter()

    old_ans = old_pred["answer"]
    old_ev = old_pred["evidence"]

    # Decision logic — KEEP old evidence (selector is 88.9%); only improve answers
    final_ev = old_ev
    if new_ans and new_ans == old_ans:
        # Both agree — high confidence, keep answer
        final_ans = old_ans
        decision = "agree"
    elif new_ans and ans_votes.get(new_ans, 0) >= 8:
        # New answer has very strong consensus (8+/10) — trust it
        final_ans = new_ans
        decision = "new_strong"
    else:
        # Disagree or weak consensus — keep old answer (87.6% baseline)
        final_ans = old_ans
        decision = "keep_old"

    return {
        "instance_id": tid,
        "answer": final_ans,
        "evidence": final_ev,
        "meta": {
            "decision": decision,
            "old_ans": old_ans, "new_ans": new_ans,
            "old_ev": old_ev, "new_ev": [evidence_id] if new_ev else None,
            "ev_votes": dict(ev_votes), "ans_votes": dict(ans_votes),
        }
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--predictions", default="data/predictions/val/predictions.jsonl")
    ap.add_argument("--llm-url", default="http://localhost:8000/v1")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ev-samples", type=int, default=5)
    ap.add_argument("--ans-samples", type=int, default=10)
    args = ap.parse_args()

    client = LLMClient(base_url=args.llm_url, model=args.llm_model)

    # Load old predictions
    preds = {}
    with open(args.predictions) as f:
        for line in f:
            d = json.loads(line)
            preds[d["instance_id"]] = d

    tasks = split_tasks(args.split, Path("data"))
    if args.limit:
        tasks = tasks[:args.limit]

    results = []
    stats = Counter()

    def work(task):
        tid = task["instance_id"]
        bp = Path(f"data/blocks/{args.split}/{tid}.json")
        if not bp.exists() or tid not in preds:
            return None
        blocks = load_blocks(str(bp))
        return process_task(client, task, blocks, preds[tid])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            result = fut.result()
            if result is None:
                continue
            results.append(result)
            meta = result["meta"]
            stats[meta["decision"]] += 1
            done += 1
            ev_changed = meta["old_ev"] != result["evidence"]
            ans_changed = meta["old_ans"] != result["answer"]
            if done % 25 == 0 or ev_changed or ans_changed:
                tag = ""
                if ev_changed: tag += " EV_CHANGED"
                if ans_changed: tag += " ANS_CHANGED"
                print(f"  [{done}/{len(tasks)}] {result['instance_id']} "
                      f"ans={result['answer']} ev={result['evidence']} "
                      f"({meta['decision']}){tag}")

    # Save
    out_path = f"data/predictions/{args.split}/predictions_improved.jsonl"
    with open(out_path, "w") as f:
        for r in sorted(results, key=lambda x: x["instance_id"]):
            f.write(json.dumps({
                "instance_id": r["instance_id"],
                "answer": r["answer"],
                "evidence": r["evidence"],
            }) + "\n")

    print(f"\nDecisions: {dict(stats)}")
    print(f"Saved {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()
