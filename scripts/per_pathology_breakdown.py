import csv
import json
from collections import defaultdict
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
    with open(path) as f:
        return [json.loads(line) for line in f]

def safe_div(a, b):
    return a / b if b else 0.0

def get_original_path(model_key):
    p1 = RESULTS / f"{model_key}_positive_original_results.jsonl"
    p2 = RESULTS / f"{model_key}_original_results.jsonl"
    return p1 if p1.exists() else p2

def main():
    rows_out = []

    for model_key in MODEL_KEYS:
        original_rows = load_jsonl(get_original_path(model_key))

        original_map = {
            (r["image_id"], r["pathology"]): norm(r["prediction"])
            for r in original_rows
        }

        pathologies = sorted(set(r["pathology"] for r in original_rows))

        for pathology in pathologies:
            row = {
                "model_key": model_key,
                "model_name": MODEL_NAMES[model_key],
                "pathology": pathology,
            }

            orig_subset = [r for r in original_rows if r["pathology"] == pathology]
            total = len(orig_subset)
            original_yes = sum(1 for r in orig_subset if norm(r["prediction"]) == "yes")
            original_no = sum(1 for r in orig_subset if norm(r["prediction"]) == "no")

            row["total"] = total
            row["original_yes"] = original_yes
            row["original_no"] = original_no
            row["original_yes_rate"] = safe_div(original_yes, total)

            for cond in CONDITIONS:
                cond_path = RESULTS / f"{model_key}_{cond}_results.jsonl"
                cond_rows = [
                    r for r in load_jsonl(cond_path)
                    if r["pathology"] == pathology
                ]

                yes_to_no = 0
                yes_to_yes = 0
                no_to_yes = 0
                no_to_no = 0
                other = 0

                for r in cond_rows:
                    key = (r["image_id"], r["pathology"])
                    o = original_map.get(key, "missing")
                    m = norm(r["prediction"])

                    if o == "yes" and m == "no":
                        yes_to_no += 1
                    elif o == "yes" and m == "yes":
                        yes_to_yes += 1
                    elif o == "no" and m == "yes":
                        no_to_yes += 1
                    elif o == "no" and m == "no":
                        no_to_no += 1
                    else:
                        other += 1

                row[f"{cond}_yes_to_no"] = yes_to_no
                row[f"{cond}_yes_to_yes"] = yes_to_yes
                row[f"{cond}_no_to_yes"] = no_to_yes
                row[f"{cond}_no_to_no"] = no_to_no
                row[f"{cond}_other"] = other
                row[f"{cond}_delta_flip"] = safe_div(yes_to_no, original_yes)
                row[f"{cond}_artifact_flip"] = safe_div(no_to_yes, original_no)

            rows_out.append(row)

    out = RESULTS / "per_pathology_breakdown.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print("saved ->", out)

    print("\nPreview: Matched-Patch ΔFlip by model/pathology")
    for r in rows_out[:20]:
        print(
            f'{r["model_name"]:<20} '
            f'{r["pathology"]:<30} '
            f'n={r["total"]:<4} '
            f'orig_yes={r["original_yes"]:<4} '
            f'MP_Delta={r["matched_patch_delta_flip"]:.4f}'
        )

if __name__ == "__main__":
    main()
