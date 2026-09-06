"""Literal tracing: map Python code numeric literals back to source blocks.

After the solver generates and executes Python code, we trace which numeric
constants came from which evidence blocks.  This provides:
  1. A verification signal — literals should come from selected evidence
  2. Evidence pruning — blocks with no traced literals may be unnecessary
  3. A source attribution for each computation step
"""
from __future__ import annotations

import re
from typing import Any

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_COMMENT_BLOCK_RE = re.compile(r"#\s*(b\d{2})\b", re.IGNORECASE)


def extract_code_literals(code: str) -> list[str]:
    """Extract all numeric literals from Python code, ignoring comments."""
    literals = []
    for line in code.splitlines():
        # Remove comment
        code_part = line.split("#")[0]
        literals.extend(_NUM_RE.findall(code_part))
    return literals


def extract_inline_attributions(code: str) -> dict[str, list[str]]:
    """Extract inline block attributions from code comments.

    Expects the solver to annotate lines like:
        revenue = 2452.6  # b14
        costs = 180.3     # b13

    Returns: {block_id: [literal1, literal2, ...]}
    """
    attributions: dict[str, list[str]] = {}
    for line in code.splitlines():
        block_match = _COMMENT_BLOCK_RE.search(line)
        if block_match:
            block_id = block_match.group(1).lower()
            # Get literals from the code part
            code_part = line.split("#")[0]
            nums = _NUM_RE.findall(code_part)
            if nums:
                attributions.setdefault(block_id, []).extend(nums)
    return attributions


def trace_literals_to_blocks(
    code: str,
    blocks: list[dict],
) -> dict[str, list[str]]:
    """Map code numeric literals to their source blocks by text matching.

    For each numeric literal in the code, check which block(s) contain
    that exact number.  Returns {block_id: [matched_literals]}.
    """
    literals = extract_code_literals(code)
    if not literals:
        return {}

    # Build a map: literal -> set of block IDs containing it
    block_texts = {b["id"]: b["text"] for b in blocks}
    traces: dict[str, list[str]] = {}

    for lit in literals:
        # Skip trivial literals (0, 1, 2, 100) as they're too common
        try:
            val = float(lit)
            if val in (0, 1, 2, 100) or abs(val) > 1e10:
                continue
        except ValueError:
            continue

        for bid, text in block_texts.items():
            if re.search(rf"(?<!\d){re.escape(lit)}(?!\d)", text):
                traces.setdefault(bid, []).append(lit)

    return traces


def compute_trace_coverage(
    code: str,
    selected_evidence: list[str],
    blocks: list[dict],
) -> dict[str, Any]:
    """Compute how well the selected evidence covers the code's literals.

    Returns:
        {
            "total_literals": int,
            "traced_literals": int,
            "coverage": float (0-1),
            "traced_blocks": list[str],   # blocks that sourced literals
            "untraced_blocks": list[str],  # selected but no literals from them
            "missing_blocks": list[str],   # not selected but have literals
        }
    """
    literals = extract_code_literals(code)
    traces = trace_literals_to_blocks(code, blocks)
    selected_set = set(selected_evidence)

    traced_blocks = set(traces.keys())
    traced_literal_count = sum(len(v) for bid, v in traces.items() if bid in selected_set)

    # Blocks that were selected but contributed no literals
    untraced = [bid for bid in selected_evidence if bid not in traced_blocks]
    # Blocks NOT selected but that have matching literals
    missing = [bid for bid in traced_blocks if bid not in selected_set]

    # Coverage: fraction of non-trivial literals that trace to a selected block
    total = len([l for l in literals if float(l) not in (0, 1, 2, 100)])
    traced = sum(len(v) for bid, v in traces.items() if bid in selected_set)

    return {
        "total_literals": total,
        "traced_literals": traced,
        "coverage": traced / max(total, 1),
        "traced_blocks": sorted(traced_blocks & selected_set),
        "untraced_blocks": untraced,
        "missing_blocks": missing,
    }


def filter_unnecessary_evidence(
    selected_evidence: list[str],
    code: str,
    blocks: list[dict],
) -> list[str]:
    """Remove evidence blocks that contributed no literals to the code.

    Keeps blocks that:
    1. Had at least one literal traced to them, OR
    2. Are the only selected block (always keep at least one)
    """
    if len(selected_evidence) <= 1:
        return selected_evidence

    traces = trace_literals_to_blocks(code, blocks)
    traced_set = set(traces.keys())

    # Keep blocks that contributed literals
    necessary = [bid for bid in selected_evidence if bid in traced_set]

    # If none traced, keep all (conservative)
    return necessary if necessary else selected_evidence
