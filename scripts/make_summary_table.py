import csv
import json
from pathlib import Path

RESULTS = Path("results")

MODEL_KEYS = [
    "qwen25vl",
    "qwen3vl",
    "huatuogpt_vision",
    "medgemma",
    "internvl3",
    "llava_med",
]

MODEL_NAMES = {
    "qwen25vl": "Qwen2.5-VL",
    "qwen3vl": "Qwen3-VL",
    "huatuogpt_vision": "HuatuoGPT-Vision",
    "medgemma": "MedGemma",
    "internvl3": "InternVL3",
    "llava_med": "LLaVA-Med",
}

CONDITIONS = ["matched_patch", "zero", "blur"]

def norm(x):
    x = str(x).strip().lower()
    if x.startswith("yes"):
        return "yes"
    if x.startswith("no"):
        return "no"
    return "other"

def load_jsonl(path):
    return [json.loads(line) for line in open(path)]

def initial_stats(model_key):
    rows = load_jsonl(RESULTS / f"{model_key}_original_results.jsonl")
    total = len(rows)
    yes = sum(norm(r["prediction"]) == "yes" for r in rows)
    no = sum(norm(r["prediction"]) == "no" for r in rows)
    other = total - yes - no
    correct = sum(norm(r["prediction"]) == norm(r["answer"]) for r in rows)
    return {
        "total": total,
        "original_yes": yes,
        "original_no": no,
        "original_other": other,
        "original_yes_rate": yes / total if total else 0,
        "original_accuracy": correct / total if total else 0,
        "original_map": {
            (r["image_id"], r["pathology"]): norm(r["prediction"])
            for r in rows
        },
    }

def condition_stats(model_key, condition, original_map):
    rows = load_jsonl(RESULTS / f"{model_key}_{condition}_results.jsonl")
    total = len(rows)

    yes_to_no = 0
    yes_to_yes = 0
    no_to_yes = 0
    no_to_no = 0
    masked_yes = 0
    masked_no = 0
    masked_other = 0
    masked_correct = 0

    original_yes = sum(1 for v in original_map.values() if v == "yes")
    original_no = sum(1 for v in original_map.values() if v == "no")

    for r in rows:
        key = (r["image_id"], r["pathology"])
        o = original_map.get(key, "missing")
        m = norm(r["prediction"])

        if m == "yes":
            masked_yes += 1
        elif m == "no":
            masked_no += 1
        else:
            masked_other += 1

        if m == norm(r["answer"]):
            masked_correct += 1

        if o == "yes" and m == "no":
            yes_to_no += 1
        elif o == "yes" and m == "yes":
            yes_to_yes += 1
        elif o == "no" and m == "yes":
            no_to_yes += 1
        elif o == "no" and m == "no":
            no_to_no += 1

    delta_flip = yes_to_no / original_yes if original_yes else 0
    artifact_flip = no_to_yes / original_no if original_no else 0

    return {
        f"{condition}_accuracy": masked_correct / total if total else 0,
        f"{condition}_yes_rate": masked_yes / total if total else 0,
        f"{condition}_yes_to_no": yes_to_no,
        f"{condition}_no_to_yes": no_to_yes,
        f"{condition}_delta_flip": delta_flip,
        f"{condition}_artifact_flip": artifact_flip,
    }

def main():
    rows_out = []

    for model_key in MODEL_KEYS:
        init = initial_stats(model_key)
        row = {
            "model_key": model_key,
            "model_name": MODEL_NAMES[model_key],
            "total": init["total"],
            "original_accuracy": init["original_accuracy"],
            "original_yes": init["original_yes"],
            "original_no": init["original_no"],
            "original_other": init["original_other"],
            "original_yes_rate": init["original_yes_rate"],
        }

        for cond in CONDITIONS:
            row.update(condition_stats(model_key, cond, init["original_map"]))

        rows_out.append(row)

    out = RESULTS / "summary_table.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print("saved ->", out)

    print("\nModel, Acc, MP ΔFlip, Zero ΔFlip, Blur ΔFlip")
    for r in rows_out:
        print(
            f'{r["model_name"]}, '
            f'{r["original_accuracy"]:.4f}, '
            f'{r["matched_patch_delta_flip"]:.4f}, '
            f'{r["zero_delta_flip"]:.4f}, '
            f'{r["blur_delta_flip"]:.4f}'
        )

if __name__ == "__main__":
    main()
