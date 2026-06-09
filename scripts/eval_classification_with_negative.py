import argparse
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

def eval_model(model_key, dataset_prefix=""):
    if dataset_prefix:
        pos_path = RESULTS / f"{model_key}_{dataset_prefix}_original_results.jsonl"
        neg_path = RESULTS / f"{model_key}_{dataset_prefix}_negative_original_results.jsonl"
    else:
        pos_path = RESULTS / f"{model_key}_positive_original_results.jsonl"
        if not pos_path.exists():
            pos_path = RESULTS / f"{model_key}_original_results.jsonl"
        neg_path = RESULTS / f"{model_key}_negative_original_results.jsonl"

    pos = load_jsonl(pos_path)
    neg = load_jsonl(neg_path)

    tp = sum(norm(r["prediction"]) == "yes" for r in pos)
    fn = sum(norm(r["prediction"]) == "no" for r in pos)
    pos_other = len(pos) - tp - fn

    tn = sum(norm(r["prediction"]) == "no" for r in neg)
    fp = sum(norm(r["prediction"]) == "yes" for r in neg)
    neg_other = len(neg) - tn - fp

    total = len(pos) + len(neg)

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "model_key": model_key,
        "model_name": MODEL_NAMES[model_key],
        "positive_n": len(pos),
        "negative_n": len(neg),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "pos_other": pos_other,
        "neg_other": neg_other,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_positive_rate": 1 - specificity,
        "false_negative_rate": 1 - recall,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-prefix", default="", help="e.g. chestxdet10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [eval_model(m, args.dataset_prefix) for m in MODEL_KEYS]

    if args.out:
        out = Path(args.out)
    elif args.dataset_prefix:
        out = RESULTS / f"{args.dataset_prefix}_classification_summary_with_negative.csv"
    else:
        out = RESULTS / "classification_summary_with_negative.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("saved ->", out)

    print("\nModel, Acc, Precision, Recall, Specificity, F1")
    for r in rows:
        print(
            f'{r["model_name"]}, '
            f'{r["accuracy"]:.4f}, '
            f'{r["precision"]:.4f}, '
            f'{r["recall"]:.4f}, '
            f'{r["specificity"]:.4f}, '
            f'{r["f1"]:.4f}'
        )

if __name__ == "__main__":
    main()
