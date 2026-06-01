# CSE 151B Competition Final Submission

This repository contains the final reproducible inference pipeline for the CSE 151B Kaggle competition.

## Submission Checklist

- Submit this public GitHub repository link to Gradescope.
- Add all group members to the same Gradescope submission.
- Final entry point: [run_inference.py](run_inference.py).
- Final output path: [results/submission.csv](results/submission.csv).

## GPU and Runtime

- GPU used: `NVIDIA L4` with 23 GB VRAM.
- Final model: `Qwen/Qwen3-4B-Thinking-2507`.
- Inference engine: vLLM with bitsandbytes quantization.
- Final generation settings:
  - `chunk_size=8`
  - `max_tokens=4096`
  - `gpu_util=0.55`
  - `temperature=0.6`
  - `top_p=0.95`
  - `top_k=20`
  - `min_p=0.0`
  - `repetition_penalty=1.0`
- Measured runtime: `7:00:39`for the produced 943 total rows.
- 
To measure the runtime of a clean run:

```bash
time python3 run_inference.py
```

## Model Weights Setup

The final submission pipeline uses the public HuggingFace base model directly:

```text
Qwen/Qwen3-4B-Thinking-2507
```

No fine-tuned checkpoint is required to reproduce the submitted pipeline. The local [checkpoints/](checkpoints/) directory contains experimental LoRA adapters, but they are not loaded by the final `run_inference()` path.

Install the runtime dependencies in your Python environment:

```bash
pip install vllm transformers tqdm sympy numpy bitsandbytes
```

If your environment requires HuggingFace authentication for model download, run:

```bash
huggingface-cli login
```

The model id is configured in [experiments/sft_common.py](experiments/sft_common.py).

## Single Entry Point

The required single entry point is:

```python
from run_inference import run_inference

run_inference()
```

By default, this reads [data/private.jsonl](data/private.jsonl), generates all missing rows, applies post-processing, and writes [results/submission.csv](results/submission.csv).

Equivalent explicit call:

```python
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",
    out_csv="results/submission.csv",
    out_json="results/submission.json",
    work_jsonl="results/submission_work.jsonl",
    chunk_size=8,
    max_tokens=4096,
    gpu_util=0.55,
)
```

Equivalent CLI call:

```bash
python3 experiments/generate_submission_vllm.py \
    --data data/private.jsonl \
    --work-jsonl results/submission_work.jsonl \
    --out-csv results/submission.csv \
    --out-json results/submission.json
```

## Pipeline Details

[experiments/generate_submission_vllm.py](experiments/generate_submission_vllm.py) performs the full end-to-end pipeline:

- Loads `Qwen/Qwen3-4B-Thinking-2507` with vLLM.
- Builds prompts with [experiments/run_subset_experiments.py](experiments/run_subset_experiments.py).
- Runs generation on the requested dataset.
- Applies answer normalization and format cleaning with `choose_answer()` from [experiments/postprocess_results.py](experiments/postprocess_results.py).
- Writes the final CSV.

Output files:

- [results/submission_work.jsonl](results/submission_work.jsonl): resumable working file with `raw_response` and cleaned `response`.
- [results/submission.csv](results/submission.csv): final two-column Kaggle file with columns `id,response`.
- `results/submission.json`: JSON copy of the final cleaned responses.

Calling `run_inference()` is sufficient to regenerate the final CSV from the dataset and model weights; no manual post-processing is required.
