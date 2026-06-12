import argparse
import json
from pathlib import Path
from collections import Counter


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


def transition(orig, cf):
    if orig == "yes" and cf == "no":
        return "yes_to_no"
    if orig == "yes" and cf == "yes":
        return "yes_to_yes"
    if orig == "no" and cf == "no":
        return "no_to_no"
    if orig == "no" and cf == "yes":
        return "no_to_yes"
    return "other"


def ensemble_state(transitions):
    vals = list(transitions.values())

    if all(v == "yes_to_no" for v in vals):
        return "strongly_grounded_yes"

    if all(v == "yes_to_yes" for v in vals):
        return "strongly_shortcut_yes"

    if all(v == "no_to_no" for v in vals):
        return "strongly_stable_no"

    if all(v == "no_to_yes" for v in vals):
        return "strongly_unstable_no"

    if "yes_to_no" in vals and "yes_to_yes" in vals:
        return "mixed_yes"

    if "no_to_no" in vals and "no_to_yes" in vals:
        return "mixed_no"

    return "mixed_or_unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--matched-patch", required=True)
    ap.add_argument("--zero", required=True)
    ap.add_argument("--blur", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    original = load_jsonl(args.original)
    mp = {key_of(r): r for r in load_jsonl(args.matched_patch)}
    zero = {key_of(r): r for r in load_jsonl(args.zero)}
    blur = {key_of(r): r for r in load_jsonl(args.blur)}

    out = []
    missing = 0

    for r in original:
        k = key_of(r)
        if k not in mp or k not in zero or k not in blur:
            missing += 1
            continue

        orig_pred = norm_pred(r.get("prediction", r.get("raw_prediction", "")))

        preds = {
            "matched_patch": norm_pred(mp[k].get("prediction", mp[k].get("raw_prediction", ""))),
            "zero": norm_pred(zero[k].get("prediction", zero[k].get("raw_prediction", ""))),
            "blur": norm_pred(blur[k].get("prediction", blur[k].get("raw_prediction", ""))),
        }

        trans = {op: transition(orig_pred, pred) for op, pred in preds.items()}
        state = ensemble_state(trans)

        out.append({
            "image_id": r.get("image_id"),
            "image_path": r.get("image_path"),
            "pathology": r.get("pathology"),
            "dataset": r.get("dataset", ""),
            "model": r.get("model", ""),
            "answer": norm_pred(r.get("answer", "")),
            "bbox": r.get("bbox"),
            "question": r.get("question"),
            "original_prediction": orig_pred,
            "matched_patch_prediction": preds["matched_patch"],
            "zero_prediction": preds["zero"],
            "blur_prediction": preds["blur"],
            "matched_patch_transition": trans["matched_patch"],
            "zero_transition": trans["zero"],
            "blur_transition": trans["blur"],
            "ensemble_state": state,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    c = Counter(r["ensemble_state"] for r in out)
    total = len(out)

    summary = {
        "dataset": out[0].get("dataset", "") if out else "",
        "model": out[0].get("model", "") if out else "",
        "total": total,
        "missing": missing,
        "strongly_grounded_yes": c["strongly_grounded_yes"],
        "strongly_shortcut_yes": c["strongly_shortcut_yes"],
        "strongly_stable_no": c["strongly_stable_no"],
        "strongly_unstable_no": c["strongly_unstable_no"],
        "mixed_yes": c["mixed_yes"],
        "mixed_no": c["mixed_no"],
        "mixed_or_unknown": c["mixed_or_unknown"],
    }

    for k in list(summary.keys()):
        if k in ["dataset", "model", "total", "missing"]:
            continue
        summary[k + "_rate"] = summary[k] / total if total else 0

    import csv
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"saved -> {args.output}")
    print(f"summary -> {args.summary}")
    print(f"matched pairs = {total}")
    print(f"missing pairs = {missing}")
    for k, v in summary.items():
        if k.endswith("_rate"):
            print(f"{k}: {v:.4f}")
        elif k not in ["dataset", "model"]:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
