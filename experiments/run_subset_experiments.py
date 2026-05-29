import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

sys.path.insert(0, ".")
from judger import Judger


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
LETTERS = "ABCDEFGHIJ"


BASELINE_SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

BASELINE_SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

EXACT_SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}. "
    "Prefer exact symbolic answers. Do not round unless the problem explicitly asks you to round. "
    "If a decimal is required, give at least 12 significant digits."
)

EXACT_SYSTEM_PROMPT_MCQ = BASELINE_SYSTEM_PROMPT_MCQ

SLOT_SYSTEM_PROMPT_FREE = (
    "Solve the math problem carefully. Show the reasoning needed to avoid mistakes, but be concise. "
    "Do not stop before giving the final answer. "
    "For numeric answers, do not round unless the problem asks you to round. "
    "Put the final answer inside one \\boxed{...}. "
    "For multiple blanks, put comma-separated answers inside one \\boxed{...} in the same order."
)

SLOT_SYSTEM_PROMPT_MCQ = (
    "Solve the multiple-choice math problem carefully. Choose exactly one option. "
    "End with only one boxed letter, e.g. \\boxed{C}."
)

DIRECT_SYSTEM_PROMPT_FREE = (
    "Solve the problem. Keep the work short. "
    "Return only the final answer inside \\boxed{...}. "
    "If there are multiple blanks, return one comma-separated \\boxed{...} list in order."
)

DIRECT_SYSTEM_PROMPT_MCQ = (
    "Choose the best answer. Return only one boxed option letter, e.g. \\boxed{C}."
)

STRICT_SYSTEM_PROMPT_FREE = (
    "Solve the problem internally, but output only the final answer. "
    "Use exactly one \\boxed{...}. No explanation after the box. "
    "Prefer exact symbolic expressions over decimals unless the problem explicitly asks for a decimal. "
    "If a decimal is required, give at least 12 significant digits unless rounding is requested. "
    "For multiple blanks, put comma-separated answers inside the same \\boxed{...} in order."
)

STRICT_SYSTEM_PROMPT_MCQ = (
    "Solve the multiple-choice problem internally, then output only one boxed letter, e.g. \\boxed{C}. "
    "No explanation after the box."
)


def options_text(options):
    labels = [chr(65 + i) for i in range(len(options))]
    return "\n".join(f"{label}. {option.strip()}" for label, option in zip(labels, options))


def build_baseline_prompt(item):
    if item.get("options"):
        return (
            BASELINE_SYSTEM_PROMPT_MCQ,
            f"{item['question']}\n\nOptions:\n{options_text(item['options'])}",
        )
    return BASELINE_SYSTEM_PROMPT_MATH, item["question"]


def build_exact_prompt(item):
    if item.get("options"):
        return (
            EXACT_SYSTEM_PROMPT_MCQ,
            f"{item['question']}\n\nOptions:\n{options_text(item['options'])}",
        )
    return EXACT_SYSTEM_PROMPT_MATH, item["question"]


def build_slot_prompt(item):
    q = item["question"]
    slots = q.count("[ANS]")
    if item.get("options"):
        return (
            SLOT_SYSTEM_PROMPT_MCQ,
            f"{q}\n\nOptions:\n{options_text(item['options'])}\n\nReturn one boxed option letter.",
        )
    if slots > 1:
        reminder = f"There are {slots} [ANS] blanks. Return exactly {slots} answers in order."
    elif slots == 1:
        reminder = "There is 1 [ANS] blank. Return exactly 1 answer."
    else:
        reminder = "Return the requested final answer."
    return SLOT_SYSTEM_PROMPT_FREE, f"{q}\n\n{reminder}"


def build_direct_prompt(item):
    q = item["question"]
    slots = q.count("[ANS]")
    if item.get("options"):
        return (
            DIRECT_SYSTEM_PROMPT_MCQ,
            f"{q}\n\nOptions:\n{options_text(item['options'])}",
        )
    if slots > 1:
        q = f"{q}\n\nReturn exactly {slots} answers in order."
    return DIRECT_SYSTEM_PROMPT_FREE, q


def build_direct_nothink_prompt(item):
    system, user = build_direct_prompt(item)
    return system, f"{user}\n\n/no_think"


def build_slot_nothink_prompt(item):
    system, user = build_slot_prompt(item)
    return system, f"{user}\n\n/no_think"


def build_strict_prompt(item):
    q = item["question"]
    slots = q.count("[ANS]")
    if item.get("options"):
        return STRICT_SYSTEM_PROMPT_MCQ, f"{q}\n\nOptions:\n{options_text(item['options'])}"
    if slots > 1:
        q = f"{q}\n\nThere are {slots} blanks. Return exactly {slots} answers in order."
    elif slots == 1:
        q = f"{q}\n\nReturn exactly 1 answer."
    return STRICT_SYSTEM_PROMPT_FREE, q


PROMPT_BUILDERS = {
    "orig": build_baseline_prompt,
    "orig_exact": build_exact_prompt,
    "slot": build_slot_prompt,
    "direct": build_direct_prompt,
    "direct_nothink": build_direct_nothink_prompt,
    "slot_nothink": build_slot_nothink_prompt,
    "strict": build_strict_prompt,
}


def make_chat(tokenizer, system, user):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_all_boxed(text):
    return re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)


def extract_final_answer(text):
    text = (text or "").replace("</think>", "").strip()
    boxed = extract_all_boxed(text)
    if boxed:
        return boxed[-1].strip()
    matches = re.findall(
        r"(?im)^\s*(?:\*\*)?\s*final\s*answer\s*(?:\*\*)?\s*:\s*(.+?)\s*$",
        text,
    )
    if matches:
        return matches[-1].strip().strip("$ ")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def extract_letter(text):
    ans = extract_final_answer(text).strip().strip("$ ")
    m = re.fullmatch(r"(?:option\s*)?([A-Ja-j])", ans)
    if m:
        return m.group(1).upper()
    m = re.search(r"\\boxed\{\s*([A-Ja-j])\s*\}", text or "")
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-J])\b", (text or "").upper())
    return matches[-1] if matches else ""


def normalize_answer_for_vote(answer):
    answer = answer.strip().strip("$ ")
    m = re.fullmatch(r"\[\s*(.+?)\s*\]", answer)
    if m:
        answer = m.group(1).strip()
    m = re.fullmatch(r"\\boxed\{\s*(.+?)\s*\}", answer)
    if m:
        answer = m.group(1).strip()
    return re.sub(r"\s+", "", answer).lower()


def boxed_response(answer):
    answer = answer.strip().strip("$ ")
    m = re.fullmatch(r"\[\s*(.+?)\s*\]", answer)
    if m:
        answer = m.group(1).strip()
    m = re.fullmatch(r"\\boxed\{\s*(.+?)\s*\}", answer)
    if m:
        answer = m.group(1).strip()
    return f"\\boxed{{{answer}}}"


def choose_response(candidates, is_mcq):
    finals = [extract_final_answer(text) for text in candidates]
    if is_mcq:
        letters = [extract_letter(text) for text in candidates]
        votes = [letter for letter in letters if letter]
        if votes:
            winner = Counter(votes).most_common(1)[0][0]
            return f"\\boxed{{{winner}}}", {"finals": finals, "votes": dict(Counter(votes))}
        return candidates[0], {"finals": finals, "votes": {}}

    keys = [normalize_answer_for_vote(answer) for answer in finals]
    nonempty = [key for key in keys if key]
    if nonempty:
        winner_key = Counter(nonempty).most_common(1)[0][0]
        winner_idx = next(i for i, key in enumerate(keys) if key == winner_key)
        return boxed_response(finals[winner_idx]), {"finals": finals, "votes": dict(Counter(nonempty))}
    return candidates[0], {"finals": finals, "votes": {}}


def score_response(judger, item, response):
    if item.get("options"):
        return extract_letter(response) == str(item["answer"]).strip().upper()
    gold = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
    try:
        return judger.auto_judge(response, gold, [[]] * len(gold))
    except Exception:
        return False


def summarize(name, results):
    total = len(results)
    correct = sum(r["correct"] for r in results)
    mcq = [r for r in results if r["is_mcq"]]
    free = [r for r in results if not r["is_mcq"]]
    def acc(rows):
        return sum(r["correct"] for r in rows) / len(rows) * 100 if rows else 0.0
    return {
        "name": name,
        "correct": correct,
        "total": total,
        "accuracy": correct / total * 100 if total else 0.0,
        "mcq_accuracy": acc(mcq),
        "free_accuracy": acc(free),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--gpu-util", type=float, default=0.42)
    parser.add_argument("--out", default="results/subset_experiments.json")
    parser.add_argument(
        "--variants",
        default="direct_greedy_1,slot_sample_3vote,orig_sample_3vote",
        help="Comma-separated subset of variants to run.",
    )
    args = parser.parse_args()

    data = [json.loads(line) for line in open("data/public.jsonl")][: args.n]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=16384,
        trust_remote_code=True,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
    )
    params_sample = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
    )
    params_greedy = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    judger = Judger(strict_extract=False)

    variants = [
        {"name": "orig_greedy_1", "builder": "orig", "samples": 1, "params": params_greedy},
        {"name": "orig_sample_1", "builder": "orig", "samples": 1, "params": params_sample},
        {"name": "orig_exact_sample_1", "builder": "orig_exact", "samples": 1, "params": params_sample},
        {"name": "orig_exact_greedy_1", "builder": "orig_exact", "samples": 1, "params": params_greedy},
        {"name": "orig_sample_3vote", "builder": "orig", "samples": 3, "params": params_sample},
        {"name": "slot_sample_3vote", "builder": "slot", "samples": 3, "params": params_sample},
        {"name": "direct_greedy_1", "builder": "direct", "samples": 1, "params": params_greedy},
        {"name": "direct_nothink_greedy_1", "builder": "direct_nothink", "samples": 1, "params": params_greedy},
        {"name": "slot_nothink_greedy_1", "builder": "slot_nothink", "samples": 1, "params": params_greedy},
        {"name": "strict_greedy_1", "builder": "strict", "samples": 1, "params": params_greedy},
    ]
    requested = {name.strip() for name in args.variants.split(",") if name.strip()}
    variants = [variant for variant in variants if variant["name"] in requested]

    all_outputs = {}
    all_summaries = []
    for variant in variants:
        prompts = []
        prompt_items = []
        builder = PROMPT_BUILDERS[variant["builder"]]
        for item in data:
            system, user = builder(item)
            for sample_idx in range(variant["samples"]):
                prompts.append(make_chat(tokenizer, system, user))
                prompt_items.append((item, sample_idx))

        print(f"\nRunning {variant['name']} with {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params=variant["params"])
        grouped = defaultdict(list)
        for (item, _), output in zip(prompt_items, outputs):
            grouped[item["id"]].append(output.outputs[0].text.strip())

        results = []
        for item in data:
            candidates = grouped[item["id"]]
            if variant["samples"] == 1:
                response = candidates[0]
                meta = {"finals": [extract_final_answer(response)], "votes": {}}
            else:
                response, meta = choose_response(candidates, bool(item.get("options")))
            correct = score_response(judger, item, response)
            results.append(
                {
                    "id": item["id"],
                    "is_mcq": bool(item.get("options")),
                    "gold": item["answer"],
                    "response": response,
                    "final": extract_final_answer(response),
                    "meta": meta,
                    "correct": correct,
                    "raw_candidates": candidates,
                }
            )

        summary = summarize(variant["name"], results)
        all_summaries.append(summary)
        all_outputs[variant["name"]] = {"summary": summary, "results": results}
        print(
            f"{summary['name']}: {summary['correct']}/{summary['total']} "
            f"({summary['accuracy']:.1f}%), MCQ {summary['mcq_accuracy']:.1f}%, "
            f"free {summary['free_accuracy']:.1f}%"
        )
        for r in results:
            print(
                f"  id={r['id']} ok={r['correct']} final={r['final']!r} gold={r['gold']!r}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summaries": all_summaries, "variants": all_outputs}, indent=2))
    print(f"\nSaved experiment details to {out}")


if __name__ == "__main__":
    main()
