#!/usr/bin/env python3
"""Stage 2: RL fine-tuning of the evidence selector with REINFORCE.

Uses the frozen PoT solver as the environment:
  1. Selector scores blocks → sample K evidence masks
  2. For each mask: solver computes answer → reward
  3. Policy gradient updates the selector

The reward combines answer accuracy, evidence F1, code validity,
and a minimality penalty.

Usage (on RunPod with vLLM running):
  python scripts/train_selector_rl.py \
      --selector models/selector \
      --llm-url http://localhost:8000/v1 \
      --epochs 3 --masks-per-task 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from docsem.blocks import load_blocks, normalize_block_id, blocks_to_text
from docsem.config import SOLVE_PROMPT
from docsem.data import blocks_path_for, load_labels, split_tasks
from docsem.llm import LLMClient
from docsem.normalize import exec_python, extract_json_object, normalize_answer
from docsem.rl_reward import compute_reward


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", default="models/selector", help="path to supervised selector")
    ap.add_argument("--output", default="models/selector_rl", help="output directory for RL-tuned selector")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--llm-url", required=True, help="vLLM base URL")
    ap.add_argument("--llm-model", default=None, help="model name (auto-detected if not set)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--masks-per-task", type=int, default=5, help="evidence masks to sample per task")
    ap.add_argument("--max-blocks", type=int, default=3, help="max evidence blocks per mask")
    ap.add_argument("--top-k-candidates", type=int, default=15, help="candidate blocks to consider")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature for masks")
    ap.add_argument("--solver-samples", type=int, default=3, help="self-consistency samples for solver")
    ap.add_argument("--eval-every", type=int, default=50, help="evaluate every N tasks")
    ap.add_argument("--dev-frac", type=float, default=0.1, help="fraction of tasks for dev")
    return ap.parse_args()


def _solve_with_blocks(
    client: LLMClient,
    query: str,
    selected_blocks: list[dict],
    samples: int = 3,
) -> dict:
    """Run the PoT solver on selected blocks and return answer + metadata."""
    if not selected_blocks:
        return {"answer": None, "evidence": [], "code_executed": False, "code_valid": False}

    prompt = SOLVE_PROMPT.format(query=query, blocks=blocks_to_text(selected_blocks))
    try:
        texts = client.chat(
            prompt,
            temperature=0.6,
            max_tokens=1024,
            n=samples,
            system="You return only valid JSON. Be precise with arithmetic.",
        )
    except Exception:
        return {"answer": None, "evidence": [], "code_executed": False, "code_valid": False}

    # Parse and execute each sample
    best = None
    for text in texts:
        obj = extract_json_object(text)
        if not obj:
            continue
        code = obj.get("python_code", "")
        stated = obj.get("answer")
        if isinstance(stated, (int, float)):
            stated = normalize_answer(str(stated))
        elif isinstance(stated, str):
            stated = normalize_answer(stated)

        exec_ans = exec_python(code) if code.strip() else None
        answer = exec_ans if exec_ans is not None else stated

        ev = obj.get("evidence", [])
        if isinstance(ev, str):
            ev = [ev]
        ev = [e for e in ev if normalize_block_id(str(e))]

        if answer is not None:
            if best is None or (exec_ans is not None and best.get("exec_ans") is None):
                best = {
                    "answer": answer,
                    "evidence": ev,
                    "code_executed": exec_ans is not None,
                    "code_valid": exec_ans is not None and stated is not None and exec_ans == stated,
                }

    return best or {"answer": None, "evidence": [], "code_executed": False, "code_valid": False}


def _compute_block_log_probs(
    model,
    tokenizer,
    query: str,
    block_texts: list[str],
    selected_indices: list[int],
    device: str,
) -> torch.Tensor:
    """Compute log-probability of a binary mask under the selector policy.

    For selected blocks: log(sigmoid(logit))
    For unselected blocks: log(1 - sigmoid(logit))
    Returns total log-prob (scalar tensor with grad).
    """
    enc = tokenizer(
        [query] * len(block_texts),
        block_texts,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    ).to(device)

    logits = model(**enc).logits  # [N, 2]
    log_odds = logits[:, 1] - logits[:, 0]  # [N]

    selected_set = set(selected_indices)
    mask = torch.zeros(len(block_texts), device=device)
    for idx in selected_indices:
        mask[idx] = 1.0

    # log P(mask) = sum_i [ mask_i * log(σ(z_i)) + (1-mask_i) * log(1-σ(z_i)) ]
    log_prob = (
        mask * F.logsigmoid(log_odds) +
        (1 - mask) * F.logsigmoid(-log_odds)
    ).sum()

    return log_prob


def _sample_mask(
    logits: list[float],
    max_blocks: int,
    temperature: float,
) -> list[int]:
    """Sample a binary evidence mask from logits using Gumbel noise."""
    t = torch.tensor(logits, dtype=torch.float32)
    gumbel = -torch.log(-torch.log(torch.rand_like(t) + 1e-8) + 1e-8)
    noisy = (t + gumbel) / max(temperature, 0.01)

    k = min(max_blocks, len(t))
    _, top_idx = torch.topk(noisy, k)

    # Keep indices with positive logit, always keep at least 1
    selected = []
    for idx in top_idx.tolist():
        if t[idx] > 0 or len(selected) == 0:
            selected.append(idx)
    return selected


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load selector
    print(f"Loading selector from {args.selector}")
    tokenizer = AutoTokenizer.from_pretrained(args.selector)
    model = AutoModelForSequenceClassification.from_pretrained(args.selector).to(device)
    model.train()

    # Load solver client
    client = LLMClient(base_url=args.llm_url, model=args.llm_model or "auto")

    # Load tasks and labels
    tasks = split_tasks("train", data_dir)
    labels = load_labels(data_dir / "train" / "labels.jsonl")

    # Split into train/dev
    task_ids = sorted(t["instance_id"] for t in tasks if t["instance_id"] in labels)
    dev_count = max(1, int(len(task_ids) * args.dev_frac))
    dev_ids = set(task_ids[:dev_count])
    train_ids = [tid for tid in task_ids if tid not in dev_ids]

    task_by_id = {t["instance_id"]: t for t in tasks}
    print(f"RL training: {len(train_ids)} tasks, {len(dev_ids)} dev tasks")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    best_dev_reward = -float("inf")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        epoch_rewards = []
        epoch_loss = 0.0

        import random
        random.shuffle(train_ids)

        for task_idx, tid in enumerate(train_ids):
            task = task_by_id[tid]
            query = task["user_query"]
            gold = labels[tid]
            gold_answer = gold["answer"]
            gold_evidence = gold["evidence"]

            # Load blocks
            bp = blocks_path_for(tid, "train", data_dir)
            if not bp.exists():
                continue
            blocks = load_blocks(bp)
            block_texts = [b["text"] for b in blocks]
            block_ids = [b["id"] for b in blocks]
            all_ids = set(block_ids)

            if not blocks:
                continue

            # Get logits for candidate selection
            with torch.no_grad():
                enc = tokenizer(
                    [query] * len(block_texts),
                    block_texts,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                raw_logits = model(**enc).logits
                log_odds = (raw_logits[:, 1] - raw_logits[:, 0]).tolist()

            # Sample K evidence masks
            masks_indices = []
            for _ in range(args.masks_per_task):
                selected = _sample_mask(log_odds, args.max_blocks, args.temperature)
                masks_indices.append(selected)

            # Evaluate each mask with the solver
            rewards = []
            for selected_idx in masks_indices:
                selected_blocks = [blocks[i] for i in selected_idx]
                selected_ids = [block_ids[i] for i in selected_idx]

                # Run solver
                result = _solve_with_blocks(client, query, selected_blocks, args.solver_samples)

                # Compute reward
                r = compute_reward(
                    pred_answer=result["answer"],
                    gold_answer=gold_answer,
                    pred_evidence=selected_ids,
                    gold_evidence=gold_evidence,
                    code_executed=result["code_executed"],
                    code_valid=result["code_valid"],
                    all_block_ids=all_ids,
                )
                rewards.append(r)

            # REINFORCE update
            baseline = sum(rewards) / len(rewards)
            epoch_rewards.extend(rewards)

            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for selected_idx, r in zip(masks_indices, rewards):
                advantage = r - baseline
                log_prob = _compute_block_log_probs(
                    model, tokenizer, query, block_texts, selected_idx, device
                )
                # REINFORCE: loss = -advantage * log_prob
                loss = -advantage * log_prob
                total_loss = total_loss + loss

            total_loss = total_loss / len(masks_indices)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total_loss.item()

            if (task_idx + 1) % 10 == 0:
                avg_r = sum(epoch_rewards[-50:]) / min(50, len(epoch_rewards[-50:]))
                print(f"  [{task_idx+1}/{len(train_ids)}] avg_reward={avg_r:.3f} loss={epoch_loss/(task_idx+1):.4f}")

            # Dev evaluation
            if (task_idx + 1) % args.eval_every == 0:
                dev_reward = _evaluate_dev(
                    model, tokenizer, client, task_by_id, labels, dev_ids,
                    data_dir, device, args,
                )
                print(f"  DEV avg_reward={dev_reward:.3f} (best={best_dev_reward:.3f})")
                if dev_reward > best_dev_reward:
                    best_dev_reward = dev_reward
                    model.save_pretrained(str(out_dir))
                    tokenizer.save_pretrained(str(out_dir))
                    print(f"  Saved new best to {out_dir}")
                model.train()

        # End-of-epoch eval
        avg_epoch = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        print(f"Epoch {epoch+1} done. avg_reward={avg_epoch:.3f} avg_loss={epoch_loss/max(len(train_ids),1):.4f}")

        dev_reward = _evaluate_dev(
            model, tokenizer, client, task_by_id, labels, dev_ids,
            data_dir, device, args,
        )
        print(f"DEV avg_reward={dev_reward:.3f} (best={best_dev_reward:.3f})")
        if dev_reward > best_dev_reward:
            best_dev_reward = dev_reward
            model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            print(f"Saved new best to {out_dir}")

    print(f"\nDone. Best dev reward: {best_dev_reward:.3f}")
    print(f"RL-tuned selector saved to {out_dir}")


def _evaluate_dev(
    model, tokenizer, client, task_by_id, labels, dev_ids,
    data_dir, device, args,
) -> float:
    """Evaluate on dev tasks using greedy selection (no sampling)."""
    model.eval()
    rewards = []

    with torch.no_grad():
        for tid in sorted(dev_ids):
            if tid not in task_by_id or tid not in labels:
                continue
            task = task_by_id[tid]
            query = task["user_query"]
            gold = labels[tid]

            bp = blocks_path_for(tid, "train", data_dir)
            if not bp.exists():
                continue
            blocks = load_blocks(bp)
            block_texts = [b["text"] for b in blocks]
            block_ids = [b["id"] for b in blocks]

            if not blocks:
                continue

            # Greedy: pick top-k blocks by score
            enc = tokenizer(
                [query] * len(block_texts),
                block_texts,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(device)
            raw_logits = model(**enc).logits
            probs = torch.softmax(raw_logits, dim=-1)[:, 1]
            _, top_idx = torch.topk(probs, min(args.max_blocks, len(probs)))

            selected_blocks = [blocks[i] for i in top_idx.tolist()]
            selected_ids = [block_ids[i] for i in top_idx.tolist()]

            result = _solve_with_blocks(client, query, selected_blocks, args.solver_samples)
            r = compute_reward(
                pred_answer=result["answer"],
                gold_answer=gold["answer"],
                pred_evidence=selected_ids,
                gold_evidence=gold["evidence"],
                code_executed=result["code_executed"],
                code_valid=result["code_valid"],
                all_block_ids=set(block_ids),
            )
            rewards.append(r)

    return sum(rewards) / max(len(rewards), 1)


if __name__ == "__main__":
    main()
