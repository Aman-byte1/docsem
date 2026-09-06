#!/usr/bin/env bash
# One-command vLLM server setup for the DocSem pipeline (small-model stack).
#
# Starts TWO servers on one A100:
#   :8000  Qwen/Qwen2.5-VL-3B-Instruct    (vision OCR)
#   :8001  Qwen/Qwen3-4B-Instruct-2507    (text-only PoT solver)
#
# Total VRAM ~22 GB — both fit side by side on a 40 GB A100, comfortably on 80 GB.
#
# Handles every environment gotcha we hit on RunPod:
#   1. CUDA sanity check (needs a GPU pod)
#   2. Downloads both models to plain dirs if missing (~11 GB total)
#   3. Patches config.json: vLLM < 0.28 chokes when the HF config carries the
#      legacy rope_scaling.type="mrope" AND vLLM injects a modern rope_type;
#      rewrite it as the modern {"rope_type": "mrope", ...} so only one field exists
#   4. Starts both `vllm serve` processes and waits until each answers /v1/models
#
# Usage (run from the repo root on the pod):
#   bash scripts/setup_vllm.sh
#   # or: only the OCR server:
#   bash scripts/setup_vllm.sh --single
set -euo pipefail

VL_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
TXT_MODEL="Qwen/Qwen3-4B-Instruct-2507"
VL_DIR="${VL_DIR:-/workspace/qwen25vl3b}"
TXT_DIR="${TXT_DIR:-/workspace/qwen3-4b}"
PORT="${PORT:-8000}"
TXT_PORT="${TXT_PORT:-8001}"
SINGLE=0
[ "${1:-}" = "--single" ] && SINGLE=1

echo "==> [1/4] CUDA sanity check"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available - is this a GPU pod? (nvidia-smi should show an A100)"
print("GPU OK:", torch.cuda.get_device_name(0))
PY

# Authenticate once if HF_TOKEN is set (avoids 429 rate limits on downloads)
if [ -n "${HF_TOKEN:-}" ]; then
    hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi

# hf download with retry — HF rate-limits (429) are transient; resume is safe
download_model () {
    local repo="$1" dir="$2"
    for attempt in 1 2 3 4 5; do
        if hf download "$repo" --local-dir "$dir"; then
            return 0
        fi
        echo "  download attempt $attempt/5 failed (often a 429 rate limit) — waiting 60s and resuming..."
        sleep 60
    done
    echo "ERROR: could not download $repo after 5 attempts. Set HF_TOKEN and re-run."
    return 1
}

echo "==> [2/4] Model downloads (if missing)"
if [ ! -f "$VL_DIR/config.json" ]; then
    download_model "$VL_MODEL" "$VL_DIR"
else
    echo "VL model already present at $VL_DIR"
fi
if [ "$SINGLE" -eq 0 ] && [ ! -f "$TXT_DIR/config.json" ]; then
    download_model "$TXT_MODEL" "$TXT_DIR"
else
    echo "Text model already present at $TXT_DIR"
fi

patch_config () {
python - "$1" <<'PY'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
rs = cfg.get("rope_scaling")
changed = False
if rs and rs.get("type") == "mrope" and "rope_type" not in rs:
    cfg["rope_scaling"] = {"rope_type": "mrope", "mrope_section": rs["mrope_section"]}
    changed = True
# same legacy/modern conflict can appear on text_config for some Qwen releases
tc = cfg.get("text_config") or {}
if tc and tc.get("rope_scaling", {}).get("type") == "mrope" and "rope_type" not in tc.get("rope_scaling", {}):
    tc["rope_scaling"] = {"rope_type": "mrope", "mrope_section": tc["rope_scaling"]["mrope_section"]}
    changed = True
if changed:
    json.dump(cfg, open(p, "w"), indent=2)
    print("patched:", p)
else:
    print("already compatible:", p)
PY
}

echo "==> [3/4] Patch config.json (legacy mrope -> modern rope_type) for vLLM"
patch_config "$VL_DIR/config.json"
[ "$SINGLE" -eq 0 ] && patch_config "$TXT_DIR/config.json"

echo "==> [4/4] Starting vLLM servers"
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

nohup vllm serve "$VL_DIR" --port "$PORT" --max-model-len 8192 --gpu-memory-utilization 0.88 > vllm-ocr.log 2>&1 &
if [ "$SINGLE" -eq 0 ]; then
  sleep 5
  nohup vllm serve "$TXT_DIR" --port "$TXT_PORT" --max-model-len 8192 --gpu-memory-utilization 0.88 > vllm-solver.log 2>&1 &
fi

until curl -s "http://localhost:$PORT/v1/models" | grep -qi "qwen"; do sleep 5; done
echo "OCR server READY on port $PORT"
if [ "$SINGLE" -eq 0 ]; then
  until curl -s "http://localhost:$TXT_PORT/v1/models" | grep -qi "qwen"; do sleep 5; done
  echo "Solver server READY on port $TXT_PORT"
fi
echo "DONE"
