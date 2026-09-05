"""RL reward computation for the Evidence Policy Network.

Computes a scalar reward for a (predicted_answer, predicted_evidence) pair
against gold labels.  Also provides evidence-mask sampling from selector logits
for REINFORCE-style training.
"""
from __future__ import annotations

import re
from typing import Any

from .blocks import normalize_block_id
from .metrics import answer_em, evidence_f1
from .normalize import normalize_answer


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(
    pred_answer: str | None,
    gold_answer: str,
    pred_evidence: list[str],
    gold_evidence: list[str],
    code_executed: bool = False,
    code_valid: bool = False,
    all_block_ids: set[str] | None = None,
) -> float:
    """Combined reward for a single task prediction.

    Weights (from the plan):
      +3.0  answer correctness (exact match after normalization)
      +3.0  evidence block F1
      +0.5  code was successfully executed
      +0.5  valid JSON schema (answer present + evidence non-empty)
      -0.15 per unnecessary block beyond gold count
      -1.0  per invalid/non-existent block ID
    """
    r = 0.0

    # Answer correctness
    r += 3.0 * float(answer_em(pred_answer, gold_answer) if pred_answer else 0)

    # Evidence F1
    r += 3.0 * evidence_f1(pred_evidence, gold_evidence)

    # Code execution bonus
    r += 0.5 * float(code_executed)
    r += 0.5 * float(code_valid and pred_answer is not None and len(pred_evidence) > 0)

    # Minimality penalty: unnecessary blocks
    extra = max(0, len(pred_evidence) - len(gold_evidence))
    r -= 0.15 * extra

    # Invalid block ID penalty
    if all_block_ids is not None:
        invalid = sum(1 for e in pred_evidence if e not in all_block_ids)
        r -= 1.0 * invalid

    return r


# ---------------------------------------------------------------------------
# Evidence mask sampling for REINFORCE
# ---------------------------------------------------------------------------

def sample_evidence_masks(
    block_logits: list[float],
    block_ids: list[str],
    n_masks: int = 5,
    max_blocks: int = 3,
    temperature: float = 1.0,
) -> list[list[str]]:
    """Sample binary evidence masks from selector logits.

    Each mask is a list of selected block IDs.  We use Gumbel-softmax-style
    sampling: add Gumbel noise to logits, take the top-max_blocks.

    Args:
        block_logits: raw positive-class logits per block (higher = more relevant)
        block_ids: corresponding block IDs
        n_masks: number of masks to sample
        max_blocks: maximum blocks per mask
        temperature: sampling temperature (higher = more random)

    Returns:
        List of n_masks evidence masks (each a list of block IDs)
    """
    import torch

    logits = torch.tensor(block_logits, dtype=torch.float32)

    masks: list[list[str]] = []
    for _ in range(n_masks):
        # Gumbel noise for exploration
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        noisy = (logits + gumbel) / max(temperature, 0.01)

        # Top-k selection
        k = min(max_blocks, len(logits))
        _, top_idx = torch.topk(noisy, k)

        # Only include blocks with positive logit (above threshold)
        selected = []
        for idx in top_idx.tolist():
            if logits[idx] > 0 or len(selected) == 0:  # always include at least 1
                selected.append(block_ids[idx])
        masks.append(selected)

    return masks


def mask_log_prob(
    block_logits: list[float],
    block_ids: list[str],
    selected_ids: list[str],
) -> float:
    """Log-probability of a particular evidence mask under the selector's policy.

    P(mask) = prod_{selected} sigmoid(logit_i) * prod_{not selected} (1 - sigmoid(logit_i))
    """
    import torch

    logits = torch.tensor(block_logits, dtype=torch.float32)
    selected_set = set(selected_ids)

    log_prob = 0.0
    for logit, bid in zip(logits.tolist(), block_ids):
        p = torch.sigmoid(torch.tensor(logit)).item()
        if bid in selected_set:
            log_prob += max(torch.log(torch.tensor(p)).item(), -20.0)
        else:
            log_prob += max(torch.log(torch.tensor(1.0 - p)).item(), -20.0)

    return log_prob
