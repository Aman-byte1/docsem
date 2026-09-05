# DocSem — Document-Grounded Quantitative Reasoning with Evidence Attribution

System for the **DocInsights 2026 @ EMNLP 2026 DocSem shared task**: given a scanned
PDF brief and a paraphrased user query, return the final numerical answer plus the
supporting visible PDF block ID(s).

Key facts about the task we exploit:

- The PDFs are **full-page scans** (no text layer) — the pipeline is OCR-first.
- Every content block is **printed in the document** with a visible id like `b01: <content>`.
- Training labels give the exact gold evidence block (1 block per task in all 908 labels)
  and the answer (**always an integer** in train).
- The user query paraphrases scenario keywords ("concerning percentage and foot") that
  appear **verbatim in the target block**; the target block is the one that also contains
  the complete arithmetic question.

## Architecture

```
PDF + user_query
   │  render pages (pymupdf)
   ▼
page PNGs
   │  OCR with Qwen2.5-VL (vLLM)  — or rapidocr locally
   ▼
structured blocks: [{id: "b10", text: "..."}, ...]
   │  evidence selector: DeBERTa cross-encoder trained on the 908 gold labels
   ▼
top-k candidate blocks
   │  solver: LLM writes a Python program (Program-of-Thought) over the evidence
   ▼
execute code sandboxed  →  self-consistency (N samples, majority vote)
   ▼
{answer, evidence: ["b10"]}   →  submission.jsonl
```

- **Evidence selection** = a DeBERTa cross-encoder scoring `(query, block)` — the
  supervised Stage-1 "evidence policy". Gold labels teach relevance.
- **Answer computation** = Program-of-Thought: the LLM emits `python_code`, we execute
  it in a sandbox, and the **executed result overrides the model's stated answer**.
- **Robustness** = self-consistency: sample N solutions, majority-vote the normalized
  answer, and pick the most common evidence set among the winning group.

## Repository layout

```
docsem/            shared package (pdf, blocks, llm client, selector, solver, normalize, metrics)
scripts/
  download_data.py      HF snapshot download (~1.3 GB) into data/
  render_pdfs.py        render every page to PNG
  ocr_blocks.py         OCR pages -> blocks JSON (--engine vllm | rapidocr)
  build_selector_data.py  (query, block, label) pairs from gold evidence
  train_selector.py     train the DeBERTa evidence selector + recall@k eval
  solve.py              main inference: candidates -> PoT -> self-consistency
  evaluate.py           answer EM / evidence EM / evidence F1 on train
  make_submission.py    build + validate data/submission/val/submission.jsonl
setup_runpod.sh        one-shot pod setup (deps + data download)
requirements.txt       core deps (torch is preinstalled in the pod container)
```

All heavy steps are designed for the **RunPod A100**; nothing here needs your local GPU.
(Optional local CPU dev: `pip install -r requirements-ocr-cpu.txt` gives a rapidocr
fallback and lets you run the pipeline against any OpenAI-compatible endpoint such as
LM Studio.)

---

## RunPod A100 — step-by-step commands

> Run every block below **on the pod**, one after the other, and paste me the output of
> any step that errors.

### 0. Create the instance (in the RunPod UI)

- **GPU**: A100 (40 GB or 80 GB — either works)
- **Template / container**: PyTorch 2.x (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
- **Disk**: 60 GB+ (vLLM model cache is ~16 GB)
- Expose **port 8000** (TCP) if you want me to query the server, otherwise SSH-only is fine.

### 1. Clone + setup (one command)

```bash
git clone https://github.com/Aman-byte1/docsem.git && cd docsem && bash setup_runpod.sh
```

`setup_runpod.sh` installs dependencies and downloads the dataset into `data/`
(train + val + examples, ~1.3 GB).

### 2. Start the vLLM server (Qwen2.5-VL for OCR **and** solving)

```bash
nohup vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt image=2 \
  > vllm.log 2>&1 &

# wait for it to be ready (takes a few minutes to load the weights)
until curl -s http://localhost:8000/v1/models | grep -q Qwen; do sleep 5; done
echo "vLLM ready"
```

### 3. OCR every PDF into blocks

```bash
python scripts/render_pdfs.py --split train
python scripts/ocr_blocks.py --split train --engine vllm --llm-url http://localhost:8000/v1 --workers 8
python scripts/render_pdfs.py --split val
python scripts/ocr_blocks.py --split val   --engine vllm --llm-url http://localhost:8000/v1 --workers 8
```

~3,000–4,000 page images; a few minutes on the A100. Output:
`data/blocks/{train,val}/{task_id}.json`.

### 4. Train the evidence selector (the "training" step, minutes on A100)

```bash
python scripts/build_selector_data.py
python scripts/train_selector.py --eval-recall
```

`build_selector_data.py` turns the 908 gold labels into ~7k `(query, block, label)`
pairs (positives = gold evidence; hard negatives = same-topic wrong blocks).
`train_selector.py` trains `deberta-v3-small` and reports **recall@k** on a held-out
10% of tasks — the selector is good enough if recall@8 ≈ 0.95+.

### 5. Solve the train split and check accuracy (no labels needed for val, so use train first)

```bash
python scripts/solve.py --split train --llm-url http://localhost:8000/v1 \
    --llm-model Qwen/Qwen2.5-VL-7B-Instruct --selector models/selector \
    --samples 10 --workers 8
python scripts/evaluate.py --predictions data/predictions/train/predictions.jsonl --errors 10
```

Expected: answer accuracy 95–100% (top teams are at 100%). If it's below ~90%, tell me
the `--errors` output and we tune (more samples, temperature, top-k, prompt).

### 6. Solve validation and build the submission

```bash
python scripts/solve.py --split val --llm-url http://localhost:8000/v1 \
    --llm-model Qwen/Qwen2.5-VL-7B-Instruct --selector models/selector \
    --samples 10 --workers 8
python scripts/make_submission.py --predictions data/predictions/val/predictions.jsonl
```

This writes and validates `data/submission/val/submission.jsonl` (all 217 instances,
non-empty evidence, valid `bNN` ids). Download that file and upload it at the
[DocSem submission portal](https://docinsights-workshop.github.io/docinsights-2026/).

---

## Faster solving (optional, two servers)

If you want the text-only solver (faster than the VL model), start a second vLLM server
and point `solve.py` at it:

```bash
# start with --gpu-memory-utilization 0.45 for the VL server, then:
nohup vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --max-model-len 8192 \
  --gpu-memory-utilization 0.45 > vllm-text.log 2>&1 &

python scripts/solve.py --split train --llm-url http://localhost:8001/v1 \
    --llm-model Qwen/Qwen2.5-7B-Instruct --selector models/selector ...
```

(On a 40 GB A100 both 7B models fit with ~0.45 util each; on 80 GB it's comfortable.)

## Local CPU dev (optional, for the curious)

```bash
pip install -r requirements-ocr-cpu.txt
python scripts/download_data.py --data-dir data          # or download a few PDFs by hand
python scripts/render_pdfs.py --split train --limit 3
python scripts/ocr_blocks.py --split train --engine rapidocr --limit 3
python scripts/solve.py --split train --llm-url http://localhost:1234/v1 \
    --llm-model <your-lm-studio-model> --limit 3
```

## v2 roadmap (the RL evidence policy)

Current pipeline is the supervised Stage-1 "evidence policy". Per the plan, the upgrades:

1. **Answer-aware RL fine-tune** of the selector: reward = answer correctness +
   evidence F1 + minimality penalty (counterfactual necessity filter) — the 908 labels
   give an automatic reward, no reward model needed.
2. **Best-of-N evidence masks**: generate several evidence masks, solve each, pick the
   most consistent one (executable code + agreeing answers + block-to-operand trace).
3. **LoRA fine-tune** the solver on `(query, blocks, answer, evidence)` for cheap
   one-pass inference.

## Notes

- **Data**: `amitbcp/docinsights-2026-shared-task-data` (HuggingFace mirror of the
  `oracle-samples/gsm-sem` participant release, revision with the Aug 31 training
  corrections).
- **Evaluation**: normalized exact-match on answer (primary), evidence exact block-set
  match (secondary), evidence F1 (diagnostic).
- Use the **August 31, 2026** training release; test set drops 5 days before the
  **Sept 10** deadline.