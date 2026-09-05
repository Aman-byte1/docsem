"""Trained evidence selector (DeBERTa cross-encoder).

Scores (query, block_text) pairs in [0, 1]; at inference we rank all of a
document's blocks and feed the top-k to the solver. This is the supervised
"Stage 1 evidence policy" from the plan: gold labels teach relevance.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EvidenceSelector:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir)).to(
            self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, block_texts: list[str], batch_size: int = 32) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(block_texts), batch_size):
            batch = block_texts[i : i + batch_size]
            enc = self.tokenizer(
                [query] * len(batch),
                batch,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
            scores.extend(probs)
        return scores

    def rank_blocks(self, query: str, blocks: list[dict], top_k: int | None = None) -> list[tuple[str, float]]:
        """Return [(id, score)] for all blocks, sorted desc."""
        texts = [b["text"] for b in blocks]
        scores = self.score(query, texts)
        ranked = sorted(zip([b["id"] for b in blocks], scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k] if top_k else ranked


def load_selector(model_dir: str | Path) -> EvidenceSelector | None:
    """None if the directory doesn't contain a trained checkpoint."""
    if not Path(model_dir).exists():
        return None
    if not (Path(model_dir) / "config.json").exists():
        return None
    try:
        return EvidenceSelector(model_dir)
    except Exception:  # noqa: BLE001
        return None