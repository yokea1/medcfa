import argparse
import json
from collections import defaultdict
from pathlib import Path

def norm(x):
    x = str(x).strip().lower()
    if x.startswith("yes"):
        return "yes"
    if x.startswith("no"):
        return "no"
    return "other"

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--conditions", nargs="+", default=["matched_patch", "zero", "blur"])
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    original_path = results_dir / f"{args.model_key}_original_results.jsonl"
    original = load_jsonl(original_path)

    original_map = {}
    for r in original:
        key = (r["image_id"], r["pathology"])
        original_map[key] = norm(r["prediction"])

    print("operator,total,original_yes,yes_to_no,yes_to_yes,no_to_yes,no_to_no,other_transitions,delta_flip")

    for cond in args.conditions:
        path = results_dir / f"{args.model_key}_{cond}_results.jsonl"
        rows = load_jsonl(path)

        counts = defaultdict(int)
        original_yes = 0

        for r in rows:
            key = (r["image_id"], r["pathology"])
            o = original_map.get(key, "missing")
            m = norm(r["prediction"])

            if o == "yes":
                original_yes += 1

            transition = f"{o}_to_{m}"
            counts[transition] += 1

        yes_to_no = counts["yes_to_no"]
        yes_to_yes = counts["yes_to_yes"]
        no_to_yes = counts["no_to_yes"]
        no_to_no = counts["no_to_no"]

        other = {
            k: v for k, v in counts.items()
            if k not in {"yes_to_no", "yes_to_yes", "no_to_yes", "no_to_no"}
        }

        delta_flip = yes_to_no / original_yes if original_yes else 0.0

        print(
            f"{cond},{len(rows)},{original_yes},"
            f"{yes_to_no},{yes_to_yes},{no_to_yes},{no_to_no},"
            f"{other},{delta_flip:.4f}"
        )

if __name__ == "__main__":
    main()
