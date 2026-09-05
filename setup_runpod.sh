#!/usr/bin/env bash
# One-shot setup for a RunPod A100 instance (run from the repo root).
# Assumes: PyTorch container (torch + CUDA preinstalled), python3, ~60GB disk.
set -euo pipefail

echo "==> Updating pip"
pip install --upgrade pip

echo "==> Installing project dependencies (torch is preinstalled in the container)"
pip install -r requirements.txt

echo "==> Downloading the DocSem dataset (~1.3 GB) into data/"
python scripts/download_data.py --data-dir data

echo "==> Verifying download"
python - <<'PY'
import json, os
for split in ("train", "val"):
    p = f"data/{split}/tasks.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    pdf = f"data/{rows[0]['document_pdf']}"
    print(f"{split}: {len(rows)} tasks, first pdf exists: {os.path.exists(pdf)}")
PY

echo ""
echo "Setup complete. Next steps:"
echo "  1. Start the vLLM servers (see README 'RunPod A100' section)"
echo "  2. python scripts/render_pdfs.py --split train"
echo "  3. python scripts/ocr_blocks.py --split train --engine vllm --llm-url http://localhost:8000/v1"
echo "  4. python scripts/build_selector_data.py"
echo "  5. python scripts/train_selector.py --eval-recall"
echo "  6. python scripts/solve.py --split train --llm-url http://localhost:8001/v1"
echo "  7. python scripts/evaluate.py --predictions data/predictions/train/predictions.jsonl"