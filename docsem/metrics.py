"""Evaluation metrics matching the DocSem protocol.

- answer: normalized exact match (numbers compared numerically).
- evidence: exact block-set match; evidence F1 reported as a diagnostic
  (precision/recall over predicted vs. gold block id sets).
"""
from __future__ import annotations

from .normalize import answers_equal, normalize_answer


def answer_em(pred: str | None, gold: str) -> bool:
    return pred is not None and answers_equal(pred, gold)


def evidence_exact(pred: list[str], gold: list[str]) -> bool:
    return set(pred) == set(gold)


def evidence_f1(pred: list[str], gold: list[str]) -> float:
    p, g = set(pred), set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    precision = inter / len(p)
    recall = inter / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(predictions: list[dict], labels: dict[str, dict]) -> dict:
    """predictions: list of {instance_id, answer, evidence}; labels: id -> {answer, evidence}."""
    n = len(predictions)
    if n == 0:
        return {"instances": 0, "answer_accuracy": 0.0, "evidence_exact": 0.0, "evidence_f1": 0.0}
    a_em = sum(1 for p in predictions if answer_em(p.get("answer"), labels[p["instance_id"]]["answer"]))
    e_exact = sum(1 for p in predictions if evidence_exact(p.get("evidence") or [], labels[p["instance_id"]]["evidence"]))
    e_f1 = sum(evidence_f1(p.get("evidence") or [], labels[p["instance_id"]]["evidence"]) for p in predictions)
    return {
        "instances": n,
        "answer_accuracy": a_em / n,
        "evidence_exact": e_exact / n,
        "evidence_f1": e_f1 / n,
    }


def report(metrics: dict) -> str:
    return (
        f"instances={metrics['instances']}  "
        f"answer_acc={metrics['answer_accuracy']:.4f}  "
        f"evidence_exact={metrics['evidence_exact']:.4f}  "
        f"evidence_f1={metrics['evidence_f1']:.4f}"
    )