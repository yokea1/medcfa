import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def safe_div(a, b):
    return a / b if b else 0.0

def summarize(rows):
    total = len(rows)
    c = Counter(r["ccf_state"] for r in rows)

    reliable = c["grounded_yes"] + c["stable_no"]
    needs_review = c["shortcut_yes"] + c["unstable_no"] + c["unknown"]

    return {
        "dataset": rows[0].get("dataset", "") if rows else "",
        "model": rows[0].get("model", "") if rows else "",
        "operator": rows[0].get("operator", "") if rows else "",
        "total": total,
        "grounded_yes": c["grounded_yes"],
        "shortcut_yes": c["shortcut_yes"],
        "stable_no": c["stable_no"],
        "unstable_no": c["unstable_no"],
        "unknown": c["unknown"],
        "grounded_yes_rate": safe_div(c["grounded_yes"], total),
        "shortcut_yes_rate": safe_div(c["shortcut_yes"], total),
        "stable_no_rate": safe_div(c["stable_no"], total),
        "unstable_no_rate": safe_div(c["unstable_no"], total),
        "reliable_coverage": safe_div(reliable, total),
        "needs_review_rate": safe_div(needs_review, total),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    stats = summarize(rows)

    print("CCF reliability summary")
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
