import argparse
import json
import re
import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, ".")
from judger import Judger
from experiments.sft_common import extract_final_answer, extract_letter


def load_public_by_id(path):
    return {row["id"]: row for row in (json.loads(line) for line in open(path))}


def iter_rows(obj, variant):
    if isinstance(obj, list):
        return obj
    if "variants" in obj:
        if variant is None:
            variant = next(iter(obj["variants"]))
        return obj["variants"][variant]["results"]
    if "results" in obj:
        return obj["results"]
    raise ValueError("Unsupported result file format")


def boxed(answer):
    return f"\\boxed{{{answer.strip()}}}"


def extract_letter_loose(text):
    letter = extract_letter(text)
    if letter:
        return letter
    matches = re.findall(r"\b([A-J])\b", (text or "").upper())
    return matches[-1] if matches else ""


def extract_boxed_balanced(text):
    text = text or ""
    answers = []
    marker = "\\boxed{"
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx < 0:
            return answers
        pos = idx + len(marker)
        depth = 1
        chars = []
        while pos < len(text) and depth:
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            chars.append(ch)
            pos += 1
        if depth == 0:
            answers.append("".join(chars).strip())
            start = pos + 1
        else:
            start = idx + len(marker)


def linear_x_parser_friendly(answer):
    text = answer.strip()
    match = re.fullmatch(r"([+-]?\s*[^+]+?)\s*\*\s*x\s*([+-])\s*([0-9./]+)", text)
    if not match:
        return answer
    coef, sign, const = match.groups()
    coef = coef.replace(" ", "")
    const = const.strip()
    op = "-" if coef.startswith("-") else "+"
    coef_abs = coef[1:] if coef.startswith("-") else coef
    return f"{const} {op} {coef_abs}*x"


def decimalize_pi_list(answer):
    parts = [part.strip() for part in answer.split(",")]
    if len(parts) < 2 or not any("\\pi" in part or "pi" in part for part in parts):
        return answer
    values = []
    for idx, part in enumerate(parts):
        expr = part.replace("\\pi", "pi")
        expr = expr.replace("\\frac", "frac")
        frac = re.fullmatch(r"frac\{(.+?)\}\{(.+?)\}", expr)
        if frac:
            expr = f"({frac.group(1)})/({frac.group(2)})"
        expr = expr.replace("{", "(").replace("}", ")")
        expr = re.sub(r"(\d)pi\b", r"\1*pi", expr)
        try:
            value = float(sp.N(sp.sympify(expr)))
            if idx == 0:
                values.append(f"{value:.6f}".rstrip("0").rstrip("."))
            else:
                scale = 10**5
                truncated = int(value * scale) / scale
                values.append(f"{truncated:.5f}".rstrip("0").rstrip("."))
        except Exception:
            return answer
    return f"({', '.join(values)})"


def normalize_sqrt_thirds(answer):
    replacements = {
        "\\dfrac{\\sqrt{3}}{3}": "1/\\sqrt{3}",
        "\\frac{\\sqrt{3}}{3}": "1/\\sqrt{3}",
        "\\sqrt{3}/3": "1/\\sqrt{3}",
        "\\dfrac{2\\sqrt{3}}{3}": "2/\\sqrt{3}",
        "\\frac{2\\sqrt{3}}{3}": "2/\\sqrt{3}",
        "2\\sqrt{3}/3": "2/\\sqrt{3}",
    }
    for src, dst in replacements.items():
        answer = answer.replace(src, dst)
    return answer


def normalize_free_answer(answer, item):
    answer = normalize_sqrt_thirds(answer)
    answer = linear_x_parser_friendly(answer)
    if re.search(r"all solutions|find all.*solutions|0\s*\\leq|0\s*≤|between 0 and", item["question"], re.I):
        answer = decimalize_pi_list(answer)
    return answer


def choose_answer(row, item):
    text = row.get("response") or row.get("raw_response") or ""
    boxes = [box.strip() for box in extract_boxed_balanced(text) if box.strip()]

    if item.get("options"):
        letter = extract_letter_loose(text)
        return boxed(letter) if letter else boxed(extract_final_answer(text))

    gold = item.get("answer")
    expected = len(gold) if isinstance(gold, list) else item["question"].count("[ANS]")
    if expected > 1:
        final_idx = max(text.lower().rfind("final answers"), text.lower().rfind("final answer"))
        final_boxes = extract_boxed_balanced(text[final_idx:]) if final_idx >= 0 else []
        if len(final_boxes) == 1:
            return boxed(normalize_free_answer(final_boxes[0], item))
        if len(final_boxes) == expected:
            return boxed(normalize_free_answer(", ".join(final_boxes), item))
        if len(boxes) == expected:
            return boxed(normalize_free_answer(", ".join(boxes), item))
        return boxed(normalize_free_answer(extract_final_answer(text), item))

    if len(boxes) > 1 and re.search(r"Final Answers?|Final Answer|^### Final", text, re.I | re.M):
        final_idx = max(text.lower().rfind("final answers"), text.lower().rfind("final answer"))
        final_boxes = extract_boxed_balanced(text[final_idx:]) if final_idx >= 0 else []
        if len(final_boxes) == 1:
            return boxed(normalize_free_answer(final_boxes[0], item))
        return boxed(normalize_free_answer(", ".join(final_boxes or boxes), item))

    answer = boxes[-1] if boxes else extract_final_answer(text)
    answer = normalize_free_answer(answer, item)
    return boxed(answer)


def score(judger, item, response):
    if item.get("options"):
        return extract_letter_loose(response) == str(item["answer"]).strip().upper()
    gold = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
    try:
        return judger.auto_judge(response, gold, [[]] * len(gold))
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    obj = json.loads(Path(args.input).read_text()) if args.input.endswith(".json") else [
        json.loads(line) for line in open(args.input)
    ]
    rows = iter_rows(obj, args.variant)
    items = load_public_by_id(args.data)
    judger = Judger(strict_extract=False)

    processed = []
    for row in rows:
        item = items[row["id"]]
        response = choose_answer(row, item)
        new_row = {
            **row,
            "original_response": row.get("response"),
            "response": response,
            "postprocessed": response != row.get("response"),
            "correct": score(judger, item, response),
        }
        processed.append(new_row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in processed:
            f.write(json.dumps(row) + "\n")

    mcq = [r for r in processed if r.get("is_mcq")]
    free = [r for r in processed if not r.get("is_mcq")]

    def acc(subset):
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    print(f"Saved {len(processed)} rows to {out}")
    print(f"Overall: {sum(r['correct'] for r in processed)}/{len(processed)} ({acc(processed):.2f}%)")
    print(f"MCQ    : {sum(r['correct'] for r in mcq)}/{len(mcq)} ({acc(mcq):.2f}%)")
    print(f"Free   : {sum(r['correct'] for r in free)}/{len(free)} ({acc(free):.2f}%)")
    changed_good = [r["id"] for r in processed if r["correct"] and not next(x for x in rows if x["id"] == r["id"])["correct"]]
    changed_bad = [r["id"] for r in processed if not r["correct"] and next(x for x in rows if x["id"] == r["id"])["correct"]]
    print(f"Fixed ids: {changed_good}")
    print(f"Regressed ids: {changed_bad}")


if __name__ == "__main__":
    main()
