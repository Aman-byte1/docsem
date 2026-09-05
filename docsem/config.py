"""Central configuration: paths and shared constants for the DocSem pipeline.

All paths are relative to the project root (or an absolute --data-dir override).
The HuggingFace mirror (amitbcp/docinsights-2026-shared-task-data) is a drop-in
mirror of the oracle-samples/gsm-sem participant release; `document_pdf` entries
like "train/documents/task_000001.pdf" resolve directly under DATA_DIR.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("DOCSEM_DATA_DIR", PROJECT_ROOT / "data"))

TRAIN_TASKS = DATA_DIR / "train" / "tasks.jsonl"
TRAIN_LABELS = DATA_DIR / "train" / "labels.jsonl"
VAL_TASKS = DATA_DIR / "val" / "tasks.jsonl"

PAGES_DIR = DATA_DIR / "pages"          # rendered page PNGs
BLOCKS_DIR = DATA_DIR / "blocks"        # parsed block JSON per task
PREDICTIONS_DIR = DATA_DIR / "predictions"
SUBMISSION_DIR = DATA_DIR / "submission"

SELECTOR_DEFAULT = PROJECT_ROOT / "models" / "selector"

# Default inference settings
DEFAULT_TOP_K = 8          # candidate blocks fed to the solver
DEFAULT_SAMPLES = 10       # self-consistency samples per task
DEFAULT_TEMPERATURE = 0.6
DEFAULT_CONCURRENCY = 8

OCR_PROMPT = """\
You are transcribing a scanned internal brief, page by page. Content blocks start with a \
visible block identifier such as "b01:" or "b12:".

Transcribe this page exactly as a JSON array of objects, one per block:
[{"id": "b01", "text": "the full block text"}, ...]

Rules:
- Include EVERY block identifier printed on the page, in reading order.
- Keep numbers, units, dates, and punctuation exactly as printed (e.g. "18:00", "2452.6", "6 inches").
- Preserve multi-line block content as a single string with spaces between lines.
- Ignore page furniture that is not part of a block: "Page 1", "DocSEM | training copy", "COP", "TRAINING" stamps, standalone headers/footers.
- A block identifier may look like "b03:" or "b 10:" — normalize the id to two digits (e.g. "b03", "b10").
- If the page has no block identifiers, output [].

Return ONLY the JSON array. No commentary, no markdown fences.
"""

SOLVE_PROMPT = """\
You are solving a document-grounded quantitative reasoning task.

A user asks a question about a scanned business brief. The brief's content blocks \
have been extracted; each block starts with its block id. Your job:

1. Find the block that contains the QUANTITATIVE SCENARIO matching the user's query \
topics. This block states a self-contained math question with all numbers needed. \
Usually exactly ONE block.
2. Read that block's question carefully. Identify every number given and exactly what \
is being asked: a percentage? a total? a difference? how many?
3. Write step-by-step Python code that computes the answer. The LAST line MUST be \
`result = <expression>`.

USER QUERY:
{query}

CANDIDATE BLOCKS FROM THE DOCUMENT:
{blocks}

Return ONLY a JSON object:
{{
  "evidence": ["b09"],
  "python_code": "# How many widgets remain after removing defective ones?\\ntotal_widgets = 45\\ndefective = 12\\nresult = total_widgets - defective",
  "answer": 33
}}

Rules:
- "evidence": the block id(s) with the scenario question and its numbers (usually one).
- "python_code": executable arithmetic from the evidence block's numbers.
  * The LAST line MUST be: result = <final answer expression>
  * Use descriptive variable names and comments.
  * If the question asks "what percentage", compute (part / whole) * 100.
  * If the question asks "how many" or "what is the total", compute the count.
  * Round to a whole number with round() when the question implies an integer answer.
  * No imports, no input(), no hardcoded results.
- "answer": the final numeric result (integer when possible).
"""