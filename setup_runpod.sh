#!/usr/bin/env bash
# One-shot setup for a fresh RunPod A100 pod.
#
# Handles EVERY environment issue we hit on the first pod:
#   - pins torch 2.8.0+cu128 + vllm 0.11.0 (newer vllm needs CUDA-13 drivers the pod lacks)
#   - downgrades cuDNN to 9.7.1.26 (9.10+ fails to init on this driver: CUDNN_STATUS_NOT_INITIALIZED)
#   - pins transformers 4.55.2 (4.56+ removed all_special_tokens_extended which vllm 0.11 needs)
#   - installs hf_transfer for fast HF downloads
#   - downloads the dataset (resumable, retry-safe) and starts the vLLM server
#
# Idempotent: safe to re-run — every step skips work already done.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> [1/6] pip"
python -m pip install -q --upgrade pip

echo "==> [2/6] Project dependencies"
pip install -q -r requirements.txt

if python -c "import vllm" 2>/dev/null; then
  echo "vLLM already installed: $(python -c 'import vllm; print(vllm.__version__)')"
else
  echo "==> [3/6] Installing vLLM 0.11.0 + torch 2.8.0 (cu128 — matches the pod driver)"
  pip install -q "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
    --index-url https://download.pytorch.org/whl/cu128
  pip install -q "vllm==0.11.0"
fi

echo "==> [4/6] Pinning cuDNN 9.7.1.26 + transformers 4.55.2 (required for this driver/vllm combo)"
pip install -q --no-deps "nvidia-cudnn-cu12==9.7.1.26"
pip install -q "transformers==4.55.2"
pip install -q hf_transfer

echo "==> [5/6] GPU + cuDNN sanity check"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — is this a GPU pod?"
x = torch.randn(2, 3, 14, 14, 14, device="cuda")
w = torch.randn(4, 3, 3, 3, 3, device="cuda")
torch.nn.functional.conv3d(x, w)  # the exact op that failed with the wrong cuDNN
print("GPU OK:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| cuDNN conv3d OK")
PY

echo "==> [6/6] Dataset download (~1.3 GB, resumable — re-run safe)"
python scripts/download_data.py --data-dir data

echo ""
echo "==> Starting vLLM servers (first run downloads ~11 GB of model weights)"
bash scripts/setup_vllm.sh

echo ""
echo "=============================================================="
echo " SETUP COMPLETE — OCR server on :8000, solver server on :8001"
echo " Next commands:"
echo "   python scripts/render_pdfs.py --split train"
echo "   python scripts/ocr_blocks.py --split train --engine vllm --verify \\"
echo "       --llm-url http://localhost:8000/v1 --workers 8"
echo "   python scripts/render_pdfs.py --split val"
echo "   python scripts/ocr_blocks.py --split val --engine vllm --verify \\"
echo "       --llm-url http://localhost:8000/v1 --workers 8"
echo "   python scripts/audit_blocks.py --split train   # OCR quality gate"
echo "   python scripts/build_selector_data.py"
echo "   python scripts/train_selector.py --eval-recall"
echo "   python scripts/solve.py --split train --llm-url http://localhost:8001/v1 \\"
echo "       --selector models/selector --samples 10 --workers 8"
echo "   python scripts/evaluate.py --predictions data/predictions/train/predictions.jsonl --errors 20"
echo "=============================================================="
