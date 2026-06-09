import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

RESULTS = Path("results")
OUT = RESULTS / "yes_rate_diagnostic_summary.csv"

MODELS = [
    ("qwen25vl", "Qwen2.5-VL"),
    ("qwen3vl", "Qwen3-VL"),
    ("huatuogpt_vision", "HuatuoGPT-Vision"),
    ("medgemma", "MedGemma"),
    ("internvl3", "InternVL3"),
    ("llava_med", "LLaVA-Med"),
]

DATASETS = [
    ("chexlocalize", "", ""),
    ("chestxdet10", "chestxdet10", "chestxdet10_negative"),
]

def load_jsonl(path):
    return [json.loads(x) for x in open(path)]

def is_yes(pred):
    return str(pred).strip().lower().startswith("y")

def get_paths(model_key, dataset_prefix, neg_prefix):
    if dataset_prefix == "":
        pos1 = RESULTS / f"{model_key}_positive_original_results.jsonl"
        pos2 = RESULTS / f"{model_key}_original_results.jsonl"
        pos = pos1 if pos1.exists() else pos2
        neg = RESULTS / f"{model_key}_negative_original_results.jsonl"
    else:
        pos = RESULTS / f"{model_key}_{dataset_prefix}_original_results.jsonl"
        neg = RESULTS / f"{model_key}_{neg_prefix}_original_results.jsonl"
    return pos, neg

rows = []

for dataset_name, dataset_prefix, neg_prefix in DATASETS:
    for model_key, model_name in MODELS:
        pos_path, neg_path = get_paths(model_key, dataset_prefix, neg_prefix)

        pos = load_jsonl(pos_path)
        neg = load_jsonl(neg_path)
        orig = pos + neg

        overall_yes = sum(is_yes(r["prediction"]) for r in orig) / len(orig)
        pos_yes = sum(is_yes(r["prediction"]) for r in pos) / len(pos)
        neg_yes = sum(is_yes(r["prediction"]) for r in neg) / len(neg)

        path_rates = {}
        by_path = defaultdict(list)
        for r in orig:
            by_path[r["pathology"]].append(r)

        for p, items in by_path.items():
            path_rates[p] = sum(is_yes(x["prediction"]) for x in items) / len(items)

        path_yes_var = float(np.var(list(path_rates.values()))) if path_rates else 0.0

        always_yes_flag = (neg_yes > 0.45 and path_yes_var < 0.02)

        rows.append({
            "dataset": dataset_name,
            "model_key": model_key,
            "model_name": model_name,
            "overall_yes_rate": overall_yes,
            "positive_yes_rate_recall": pos_yes,
            "negative_yes_rate_false_positive": neg_yes,
            "path_yes_variance": path_yes_var,
            "always_yes_flag": always_yes_flag,
        })

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("saved ->", OUT)

print("\nDataset | Model | Overall Yes | Positive Yes | Negative Yes | Path Var | Always-Yes?")
for r in rows:
    print(
        f'{r["dataset"]} | {r["model_name"]} | '
        f'{r["overall_yes_rate"]:.4f} | '
        f'{r["positive_yes_rate_recall"]:.4f} | '
        f'{r["negative_yes_rate_false_positive"]:.4f} | '
        f'{r["path_yes_variance"]:.4f} | '
        f'{r["always_yes_flag"]}'
    )
