import argparse, json
from pathlib import Path

def norm_pred(x):
    x = str(x).strip().lower()
    for ch in ".!?,;:":
        x = x.replace(ch, "")
    first = x.split()[0] if x.split() else ""
    if first.startswith("yes"):
        return "yes"
    if first.startswith("no"):
        return "no"
    return "other"

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def key_of(r):
    return (
        r.get("image_id"),
        r.get("pathology"),
        r.get("model", ""),
        r.get("dataset", ""),
    )

def ccf_state(orig, cf):
    if orig == "yes" and cf == "no":
        return "grounded_yes"
    if orig == "yes" and cf == "yes":
        return "shortcut_yes"
    if orig == "no" and cf == "no":
        return "stable_no"
    if orig == "no" and cf == "yes":
        return "unstable_no"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--counterfactual", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--operator", default="matched_patch")
    args = ap.parse_args()

    original = load_jsonl(args.original)
    counterfactual = load_jsonl(args.counterfactual)
    cf_map = {key_of(r): r for r in counterfactual}

    out = []
    missing = 0

    for r in original:
        k = key_of(r)
        if k not in cf_map:
            missing += 1
            continue

        c = cf_map[k]

        orig_pred = norm_pred(r.get("prediction", r.get("raw_prediction", "")))
        cf_pred = norm_pred(c.get("prediction", c.get("raw_prediction", "")))
        answer = norm_pred(r.get("answer", ""))

        state = ccf_state(orig_pred, cf_pred)

        out.append({
            "image_id": r.get("image_id"),
            "image_path": r.get("image_path"),
            "pathology": r.get("pathology"),
            "dataset": r.get("dataset", c.get("dataset", "")),
            "model": r.get("model", c.get("model", "")),
            "answer": answer,
            "bbox": r.get("bbox"),
            "question": r.get("question"),
            "operator": args.operator,
            "original_prediction": orig_pred,
            "counterfactual_prediction": cf_pred,
            "ccf_state": state,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"saved -> {args.output}")
    print(f"matched pairs = {len(out)}")
    print(f"missing pairs = {missing}")

if __name__ == "__main__":
    main()
