#!/usr/bin/env python3
"""Replace evidence in predictions with the selector's top-1 pick.

Keeps answers unchanged — only fixes evidence attribution.
"""
import json
from pathlib import Path
from docsem.blocks import load_blocks
from docsem.data import split_tasks
from docsem.selector import EvidenceSelector

selector = EvidenceSelector("models/selector")

preds = {}
with open("data/predictions/val/predictions.jsonl") as f:
    for line in f:
        d = json.loads(line)
        preds[d["instance_id"]] = d

tasks = split_tasks("val", Path("data"))
changed = 0
for task in tasks:
    tid = task["instance_id"]
    bp = Path(f"data/blocks/val/{tid}.json")
    if not bp.exists():
        continue
    blocks = load_blocks(str(bp))
    ranked = selector.rank_blocks(task["user_query"], blocks, top_k=1)
    if ranked and tid in preds:
        best = ranked[0][0]
        if preds[tid]["evidence"] != [best]:
            print(f"  {tid}: {preds[tid]['evidence']} -> [{best}]")
            preds[tid]["evidence"] = [best]
            changed += 1

print(f"\nChanged evidence for {changed}/{len(preds)} tasks")

with open("data/predictions/val/predictions_fixed.jsonl", "w") as f:
    for d in preds.values():
        f.write(json.dumps(d) + "\n")
print("Saved to predictions_fixed.jsonl")
