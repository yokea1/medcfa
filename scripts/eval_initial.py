import argparse
import csv
import json
from pathlib import Path

def norm(x):
    x = str(x).strip().lower()
    if x.startswith("yes"):
        return "yes"
    if x.startswith("no"):
        return "no"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-keys", nargs="+", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--output", default="results/initial_summary.csv")
    args = ap.parse_args()

    rows_out = []

    for model_key in args.model_keys:
        path = Path(args.results_dir) / f"{model_key}_original_results.jsonl"
        rows = [json.loads(line) for line in open(path)]

        total = len(rows)
        yes = sum(1 for r in rows if norm(r["prediction"]) == "yes")
        no = sum(1 for r in rows if norm(r["prediction"]) == "no")
        other = total - yes - no

        correct = sum(
            1 for r in rows
            if norm(r["prediction"]) == norm(r["answer"])
        )
        accuracy = correct / total if total else 0

        rows_out.append({
            "model_key": model_key,
            "total": total,
            "yes": yes,
            "no": no,
            "other": other,
            "accuracy": accuracy,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_key", "total", "yes", "no", "other", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print("saved ->", out_path)

    for r in rows_out:
        print(
            f'{r["model_key"]}: total={r["total"]}, '
            f'yes={r["yes"]}, no={r["no"]}, '
            f'other={r["other"]}, accuracy={r["accuracy"]:.4f}'
        )

if __name__ == "__main__":
    main()
