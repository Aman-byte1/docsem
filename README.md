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
   │  OCR with Qwen2.5-VL-3B (vLLM, :8000) + second-pass verification  — or rapidocr locally
   ▼
structured blocks: [{id: "b10", text: "..."}, ...]
   │  evidence selector: DeBERTa cross-encoder trained on the 908 gold labels
   ▼
top-k candidate blocks
   │  solver: Qwen3-4B (:8001) writes a Python program (Program-of-Thought)
   ▼
execute code sandboxed  →  self-consistency (N samples, majority vote)
                          →  reflection pass when samples disagree
   ▼
{answer, evidence: ["b10"]}   →  submission.jsonl
```

- **Evidence selection** = a DeBERTa cross-encoder scoring `(query, block)` — the
  supervised Stage-1 "evidence policy". Gold labels teach relevance.
- **Answer computation** = Program-of-Thought: the LLM emits `python_code`, we execute
  it in a sandbox, and the **executed result overrides the model's stated answer**.
- **Robustness** = self-consistency: sample N solutions, majority-vote the normalized
  answer; if no answer wins a strict majority, ONE **reflection pass** re-solves with
  the disagreement shown to the model (n=3, temperature 0) before voting again.
- **OCR armor** = `--verify` double-checks every page's numbers/ids against the image
  and merges the two readings; `scripts/audit_blocks.py` flags id gaps and truncated
  blocks so they get re-OCR'd before they can poison training or solving.
- **Small-model stack** = Qwen2.5-VL-**3B** for OCR + Qwen3-**4B**-Instruct-2507 for
  solving (~22 GB total VRAM — both fit a single 40 GB A100 side by side). Accuracy
  comes from the pipeline, not the parameter count.

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

### 1. Clone + setup + server start (ONE command, handles everything)

```bash
git clone https://github.com/Aman-byte1/docsem.git 2>/dev/null; cd docsem && git pull && bash setup_runpod.sh
```

`setup_runpod.sh` is **idempotent** and installs the exact working stack we validated
on the first pod (vllm 0.11.0 + torch 2.8.0+cu128 + cuDNN 9.7.1.26 + transformers
4.55.2 — newer versions break on the pod's CUDA-12.8 driver), downloads the dataset,
starts the vLLM server on :8000 and waits for readiness. Re-running it is always safe.

> On a pod where the vLLM server died mid-session, just run `bash scripts/setup_vllm.sh`
> again — it reuses the downloaded model weights.

### 2. (Already done by setup_runpod.sh — OCR server on :8000, solver server on :8001)

The LLM clients auto-detect the served model ids — no `--llm-model` needed.

### 3. OCR every PDF into blocks (with verification)

```bash
python scripts/render_pdfs.py --split train
python scripts/ocr_blocks.py --split train --engine vllm --verify --llm-url http://localhost:8000/v1 --workers 8
python scripts/render_pdfs.py --split val
python scripts/ocr_blocks.py --split val   --engine vllm --verify --llm-url http://localhost:8000/v1 --workers 8
python scripts/audit_blocks.py --split train    # quality gate: flags id gaps/truncation
```

~3,000–4,000 page images; a few minutes on the A100. Output:
`data/blocks/{train,val}/{task_id}.json`. If the audit flags tasks, re-run
`ocr_blocks.py` with `--verify` after `python scripts/audit_blocks.py --split train --fix`
deletes the bad files.

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
python scripts/solve.py --split train --llm-url http://localhost:8001/v1 \
    --selector models/selector --samples 10 --workers 8
python scripts/evaluate.py --predictions data/predictions/train/predictions.jsonl --errors 10
```

Expected: answer accuracy 95–100% (top teams are at 100%). If it's below ~90%, tell me
the `--errors` output and we tune (more samples, temperature, top-k, prompt).

### 6. Solve validation and build the submission

```bash
python scripts/solve.py --split val --llm-url http://localhost:8001/v1 \
    --selector models/selector --samples 10 --workers 8
python scripts/make_submission.py --predictions data/predictions/val/predictions.jsonl
```

This writes and validates `data/submission/val/submission.jsonl` (all 217 instances,
non-empty evidence, valid `bNN` ids). Download that file and upload it at the
[DocSem submission portal](https://docinsights-workshop.github.io/docinsights-2026/).

---

## If you only want the OCR server

```bash
bash scripts/setup_vllm.sh --single   # starts just the VL-3B OCR server on :8000
```

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