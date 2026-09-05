"""Answer normalization, tolerant JSON extraction, and sandboxed Python execution.

Train answers are always plain integers; validation may include decimals, so we
normalize numerically (int when integral, trimmed float otherwise).
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer(raw: str) -> str | None:
    """Canonical string form of an answer, or None if no number found.

    Handles: "answer: 10", "10.0", "10 %", "the answer is 140", "10." etc.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    s = re.sub(r"^(the\s+)?(final\s+)?(answer|result|value|output)\s*(is|:|=)\s*", "", s)
    s = s.strip()
    m = NUMBER_RE.search(s)
    if not m:
        return None
    num_str = m.group(0)
    try:
        val = float(num_str)
    except ValueError:
        return None
    if val.is_integer():
        return str(int(val))
    return f"{val:.10f}".rstrip("0").rstrip(".")


def answers_equal(a: str, b: str) -> bool:
    na, nb = normalize_answer(a), normalize_answer(b)
    if na is None or nb is None:
        return False
    try:
        return float(na) == float(nb)
    except ValueError:
        return na == nb


# ---------------------------------------------------------------------------
# Tolerant JSON extraction
# ---------------------------------------------------------------------------

def _find_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def extract_json_array(text: str) -> list[Any] | None:
    """Best-effort JSON array extraction (for OCR block output)."""
    raw = _find_balanced(text, "[", "]")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    # Regex fallback: pull {"id": ..., "text": ...} pairs
    pairs = re.findall(r'"id"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if pairs:
        return [{"id": i, "text": t.replace("\\n", " ")} for i, t in pairs]
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction (for solver output)."""
    raw = _find_balanced(text, "{", "}")
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Regex fallback for the three known keys
    ev = re.findall(r'"evidence"\s*:\s*\[([^\]]*)\]', text)
    code = re.search(r'"python_code"\s*:\s*("(?:\\.|[^"\\])*")', text)
    ans = re.search(r'"answer"\s*:\s*("?[^",}\s]+"?)', text)
    out: dict[str, Any] = {}
    if ev:
        ids = re.findall(r'"([^"]+)"', ev[0])
        out["evidence"] = ids
    if code:
        try:
            out["python_code"] = json.loads(code.group(1))
        except json.JSONDecodeError:
            out["python_code"] = code.group(1)
    if ans:
        out["answer"] = ans.group(1).strip('"')
    return out or None


# ---------------------------------------------------------------------------
# Sandboxed Python execution (Program-of-Thought)
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "range": range,
    "sum": sum,
    "list": list,
    "dict": dict,
    "enumerate": enumerate,
    "zip": zip,
    "print": lambda *a: None,
}

_SAFE_GLOBALS = {
    "math": math,
    "__builtins__": _SAFE_BUILTINS,
}


def exec_python(code: str, timeout: float = 5.0) -> str | None:
    """Execute short arithmetic code and return the repr of the last expression's value.

    The model is asked to end with `result = <expression>`; we read `result`.
    Returns None on any failure.
    """
    if not code or len(code) > 4000:
        return None
    if any(bad in code for bad in ("import ", "open(", "__", "eval(", "exec(")):
        return None
    ns: dict[str, Any] = {"result": None}
    try:
        if timeout > 0:
            try:
                import signal

                def _alarm(*_a):
                    raise TimeoutError("code timeout")

                signal.signal(signal.SIGALRM, _alarm)
                signal.alarm(int(timeout))
                exec(compile(code, "<docsem>", "exec"), _SAFE_GLOBALS, ns)  # noqa: S102 - sandboxed
                signal.alarm(0)
            except (ImportError, AttributeError):
                exec(compile(code, "<docsem>", "exec"), _SAFE_GLOBALS, ns)  # noqa: S102 - sandboxed (no signal on Windows)
        else:
            exec(compile(code, "<docsem>", "exec"), _SAFE_GLOBALS, ns)  # noqa: S102
    except Exception:  # noqa: BLE001
        return None
    result = ns.get("result")
    if result is None:
        return None
    if isinstance(result, bool):
        return str(result)
    if isinstance(result, (int, float)):
        return normalize_answer(str(result))
    return str(result)