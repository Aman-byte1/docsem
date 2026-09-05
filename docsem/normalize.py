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
# a number that follows an answer marker ("=", "answer", "result", "is", ":")
MARKED_NUMBER_RE = re.compile(r"(?:answer|result|value|output|is|:|=)\s*-?\d+(?:\.\d+)?", re.IGNORECASE)


def _to_canonical(num_str: str) -> str | None:
    try:
        val = float(num_str)
    except ValueError:
        return None
    if val.is_integer():
        return str(int(val))
    # 10 decimal places avoids float noise (0.1+0.2 -> "0.3")
    return f"{round(val, 10):.10f}".rstrip("0").rstrip(".")


def normalize_answer(raw: str) -> str | None:
    """Canonical string form of an answer, or None if no number found.

    Handles: "answer: 10", "10.0", "10 %", "the answer is 140", "10." etc.
    Number selection: prefer the number attached to an answer marker
    ("= 140", "answer is 140"), then the LAST number (so "10 \u00d7 14 = 140"
    yields 140, not 10), then the first.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    # 1) number(s) following an answer marker — take the last such match
    marked = MARKED_NUMBER_RE.findall(s)
    if marked:
        num_str = NUMBER_RE.search(marked[-1])
        if num_str:
            canon = _to_canonical(num_str.group(0))
            if canon is not None:
                return canon
    # 2) last number in the string
    all_nums = NUMBER_RE.findall(s)
    if not all_nums:
        return None
    return _to_canonical(all_nums[-1])


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
}

_SAFE_GLOBALS = {
    "math": math,
    "__builtins__": _SAFE_BUILTINS,
}


def _run_with_timeout(compiled, globals_: dict, ns: dict, timeout: float) -> None:
    """Cross-platform time-limited exec: signal alarm on POSIX, daemon thread elsewhere."""
    import signal

    if hasattr(signal, "SIGALRM"):
        def _alarm(*_a):
            raise TimeoutError("code timeout")

        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(max(1, timeout)))
        try:
            exec(compiled, globals_, ns)  # noqa: S102 - sandboxed
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        return
    # Windows: no SIGALRM; run in a daemon thread and give up after the timeout.
    import threading

    done = threading.Event()

    def _run():
        try:
            exec(compiled, globals_, ns)  # noqa: S102 - sandboxed
        except Exception:
            pass
        finally:
            done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if not done.wait(timeout=max(0.1, timeout)):
        raise TimeoutError("code timeout")


def exec_python(code: str, timeout: float = 5.0) -> str | None:
    """Execute short arithmetic code and return the normalized result.

    The model is asked to end with `result = <expression>`; we read `result`.
    If `result` is missing we fall back to the last printed number (some models
    write `print(result)` instead). Returns None on any failure.
    """
    if not code or len(code) > 4000:
        return None
    if any(bad in code for bad in ("import ", "open(", "__", "eval(", "exec(")):
        return None
    try:
        compiled = compile(code, "<docsem>", "exec")
    except Exception:  # noqa: BLE001 - unparseable code
        return None
    ns: dict[str, Any] = {"result": None, "__prints": []}
    globals_ = {
        **_SAFE_GLOBALS,
        "__builtins__": {
            **_SAFE_BUILTINS,
            "print": lambda *a: ns["__prints"].append(" ".join(str(x) for x in a)),
        },
    }
    try:
        if timeout > 0:
            _run_with_timeout(compiled, globals_, ns, timeout)
        else:
            exec(compiled, globals_, ns)  # noqa: S102 - sandboxed
    except Exception:  # noqa: BLE001
        return None
    result = ns.get("result")
    if result is None:
        # try the last printed line, e.g. `print(result)`
        for line in reversed(ns.get("__prints", [])):
            norm = normalize_answer(str(line))
            if norm is not None:
                return norm
        return None
    if isinstance(result, bool):
        return str(result)
    if isinstance(result, (int, float)):
        return normalize_answer(str(result))
    return str(result)