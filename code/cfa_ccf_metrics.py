import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_div(a, b):
    return a / b if b else 0.0


def compute(rows):
    n = len(rows)
    covered = [r for r in rows if r["ccf_prediction"] != "abstain"]
    abstained = [r for r in rows if r["ccf_prediction"] == "abstain"]

    y_true = [r["answer"] for r in covered]
    y_pred = [r["ccf_prediction"] for r in covered]

    tp = sum(1 for a, p in zip(y_true, y_pred) if a == "yes" and p == "yes")
    tn = sum(1 for a, p in zip(y_true, y_pred) if a == "no" and p == "no")
    fp = sum(1 for a, p in zip(y_true, y_pred) if a == "no" and p == "yes")
    fn = sum(1 for a, p in zip(y_true, y_pred) if a == "yes" and p == "no")

    coverage = safe_div(len(covered), n)
    abstain_rate = safe_div(len(abstained), n)
    selacc = safe_div(tp + tn, len(covered))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "total": n,
        "covered": len(covered),
        "abstained": len(abstained),
        "coverage": coverage,
        "abstain_rate": abstain_rate,
        "selective_accuracy": selacc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CCF JSONL")
    ap.add_argument("--output", default=None, help="Optional output CSV")
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    stats = compute(rows)

    model = rows[0].get("model", "") if rows else ""
    dataset = rows[0].get("dataset", "") if rows else ""
    operator = rows[0].get("operator", "") if rows else ""

    stats = {
        "dataset": dataset,
        "model": model,
        "operator": operator,
        **stats,
    }

    print("CCF metrics")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
            writer.writeheader()
            writer.writerow(stats)
        print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
