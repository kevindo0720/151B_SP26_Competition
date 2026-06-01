import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, ".")
from experiments.postprocess_results import choose_answer
from experiments.run_subset_experiments import build_baseline_prompt, make_chat
from experiments.sft_common import MODEL_ID


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path):
    if not Path(path).exists():
        return {}
    done = {}
    skipped = 0
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped += 1
                print(f"WARNING: skipping bad JSONL line {line_no} in {path}: {exc}")
                continue
            if "id" not in row:
                skipped += 1
                print(f"WARNING: skipping JSONL line {line_no} in {path}: missing id")
                continue
            done[row["id"]] = row
    if skipped:
        print(f"Loaded completed rows with {skipped} skipped bad line(s).")
    return done


def append_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "response": row["response"]})


def write_json(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = [{"id": row["id"], "response": row["response"]} for row in rows]
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False)


def run_inference(
    data_path="data/private.jsonl",
    out_csv="results/submission.csv",
    out_json="results/submission.json",
    work_jsonl="results/submission_work.jsonl",
    limit=0,
    start_after_id=None,
    start_index=0,
    chunk_size=8,
    max_tokens=4096,
    gpu_util=0.55,
):
    """Run the full inference pipeline end-to-end and write final JSON/CSV outputs.

    Returns:
        list[dict]: Ordered rows containing id, is_mcq, raw_response, response.
    """
    if chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    items = load_jsonl(data_path)
    if limit:
        items = items[:limit]
    generation_items = items
    if start_after_id is not None:
        matching_positions = [i for i, item in enumerate(items) if item["id"] == start_after_id]
        if not matching_positions:
            raise ValueError(f"--start-after-id {start_after_id} was not found in {data_path}")
        generation_items = items[matching_positions[0] + 1 :]
    if start_index:
        if start_index < 0:
            raise ValueError("--start-index must be >= 0")
        generation_items = generation_items[start_index:]

    done = load_done(work_jsonl)
    remaining = [item for item in generation_items if item["id"] not in done]
    print(f"Loaded {len(items)} items from {data_path}")
    if generation_items is not items:
        print(f"Generation window: {len(generation_items)} item(s)")
    print(f"Already done in work file: {len(done)} | remaining to generate: {len(remaining)}")

    if remaining:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        llm = LLM(
            model=MODEL_ID,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            enable_prefix_caching=False,
            gpu_memory_utilization=gpu_util,
            max_model_len=16384,
            trust_remote_code=True,
            max_num_seqs=chunk_size,
            max_num_batched_tokens=16384,
        )
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.0,
        )

        for start in tqdm(range(0, len(remaining), chunk_size), desc="Generating chunks"):
            batch = remaining[start : start + chunk_size]
            prompts = []
            for item in batch:
                system, user = build_baseline_prompt(item)
                prompts.append(make_chat(tokenizer, system, user))

            outputs = llm.generate(prompts, sampling_params=params)
            rows = []
            for item, output in zip(batch, outputs):
                raw = output.outputs[0].text.strip()
                row_for_postprocess = {
                    "id": item["id"],
                    "is_mcq": bool(item.get("options")),
                    "response": raw,
                    "raw_response": raw,
                }
                response = choose_answer(row_for_postprocess, item)
                rows.append(
                    {
                        "id": item["id"],
                        "is_mcq": bool(item.get("options")),
                        "raw_response": raw,
                        "response": response,
                    }
                )
            append_rows(work_jsonl, rows)
            done.update({row["id"]: row for row in rows})

    ordered_rows = [done[item["id"]] for item in items if item["id"] in done]
    write_csv(out_csv, ordered_rows)
    write_json(out_json, ordered_rows)
    print(f"Wrote {len(ordered_rows)} rows to {out_csv}")
    print(f"Wrote {len(ordered_rows)} rows to {out_json}")
    if len(ordered_rows) != len(items):
        missing = [item["id"] for item in items if item["id"] not in done]
        print(f"WARNING: missing {len(missing)} ids: {missing[:20]}")
    if len(ordered_rows) != 943:
        print(f"NOTE: output has {len(ordered_rows)} rows; expected 943 only if this is the private/test file.")
    return ordered_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="JSONL file to run, e.g. data/private.jsonl")
    parser.add_argument("--out-csv", default="results/submission.csv")
    parser.add_argument("--out-json", default="results/submission.json")
    parser.add_argument("--work-jsonl", default="results/submission_work.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--start-after-id",
        type=int,
        default=None,
        help="Only generate rows after this id in --data; useful for resuming after a known stopping point.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Only generate rows starting at this 0-based row index in --data.",
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpu-util", type=float, default=0.55)
    args = parser.parse_args()

    run_inference(
        data_path=args.data,
        out_csv=args.out_csv,
        out_json=args.out_json,
        work_jsonl=args.work_jsonl,
        limit=args.limit,
        start_after_id=args.start_after_id,
        start_index=args.start_index,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
        gpu_util=args.gpu_util,
    )


if __name__ == "__main__":
    main()
