"""Thin OpenAI-compatible client used for both vLLM (RunPod) and LM Studio (local).

Everything the pipeline needs is `chat` (text-only completions, n>1 for
self-consistency) and `chat_with_images` (page OCR with Qwen2.5-VL).
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

from openai import OpenAI


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 300.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    @staticmethod
    def _retry(fn, tries: int = 4, backoff: float = 2.0):
        last = None
        for attempt in range(tries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 - surface after retries
                last = e
                time.sleep(backoff * (2**attempt))
        raise last

    def chat(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        n: int = 1,
        system: str | None = None,
    ) -> list[str]:
        """Return n sampled completions for a text prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def _call():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            return [c.message.content or "" for c in resp.choices]

        return self._retry(_call)

    def chat_with_images(
        self,
        prompt: str,
        images: list[str | Path],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Single completion with one or more images (base64 data URLs)."""
        content: list[dict] = []
        for img in images:
            if isinstance(img, Path) or "\n" not in str(img) and Path(str(img)).exists():
                p = Path(img)
                b64 = base64.b64encode(p.read_bytes()).decode()
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            else:
                b64 = img
                mime = "image/png"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        content.append({"type": "text", "text": prompt})

        def _call():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        return self._retry(_call)


def client_from_args(base_url: str | None, model: str | None, default_model: str) -> LLMClient | None:
    """Build a client from CLI args; None if no URL given (caller decides fallback)."""
    if not base_url:
        return None
    return LLMClient(base_url=base_url, model=model or default_model)