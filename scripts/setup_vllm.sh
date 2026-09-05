#!/usr/bin/env bash
# One-command vLLM server setup for the DocSem pipeline.
#
# Handles every environment gotcha we hit on RunPod:
#   1. CUDA sanity check (needs a GPU pod)
#   2. Downloads Qwen/Qwen2.5-VL-7B-Instruct to a plain dir (~16 GB) if missing
#   3. Patches config.json: vLLM < 0.28 chokes when the HF config carries the
#      legacy rope_scaling.type="mrope" AND vLLM injects a modern rope_type;
#      rewrite it as the modern {"rope_type": "mrope", ...} so only one field exists
#   4. Starts `vllm serve` on port 8000 and waits until it answers /v1/models
#
# Usage (run from the repo root on the pod):
#   bash scripts/setup_vllm.sh
#   # or: MODEL_DIR=/path/to/model bash scripts/setup_vllm.sh
#
# Requires: python3 + torch with CUDA (preinstalled), `pip install vllm` done,
# and the vllm version compatible with your torch/driver combo (see README).
set -euo pipefail

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_DIR="${MODEL_DIR:-/workspace/qwen25vl}"
PORT="${PORT:-8000}"

echo "==> [1/4] CUDA sanity check"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available - is this a GPU pod? (nvidia-smi should show an A100)"
print("GPU OK:", torch.cuda.get_device_name(0))
PY

echo "==> [2/4] Model download (if missing) to $MODEL_DIR"
if [ ! -f "$MODEL_DIR/config.json" ]; then
    hf download "$MODEL" --local-dir "$MODEL_DIR"
else
    echo "Model already present at $MODEL_DIR"
fi

echo "==> [3/4] Patch config.json (legacy mrope -> modern rope_type) for vLLM"
python - "$MODEL_DIR/config.json" <<'PY'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
rs = cfg.get("rope_scaling")
if rs and rs.get("type") == "mrope" and "rope_type" not in rs:
    cfg["rope_scaling"] = {"rope_type": "mrope", "mrope_section": rs["mrope_section"]}
    json.dump(cfg, open(p, "w"), indent=2)
    print("patched config.json: rope_scaling =", cfg["rope_scaling"])
else:
    print("config.json already compatible: rope_scaling =", rs)
PY

echo "==> [4/4] Starting vLLM on port $PORT (log: vllm.log)"
pkill -f "vllm serve" 2>/dev/null || true
nohup vllm serve "$MODEL_DIR" --port "$PORT" --max-model-len 8192 --gpu-memory-utilization 0.9 > vllm.log 2>&1 &
until curl -s "http://localhost:$PORT/v1/models" | grep -q Qwen; do sleep 5; done
echo "VLLM READY on port $PORT"
