import argparse
import csv
import json
import random
from pathlib import Path

RESULTS = Path("results")
N_BOOT = 1000
SEED = 42

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
    return [json.loads(line) for line in open(path)]

def safe_div(a, b):
    return a / b if b else 0.0

def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = int(round((len(xs) - 1) * p))
    return xs[k]

def classification_metrics(pos_rows, neg_rows):
    tp = sum(norm(r["prediction"]) == "yes" for r in pos_rows)
    fn = sum(norm(r["prediction"]) == "no" for r in pos_rows)
    tn = sum(norm(r["prediction"]) == "no" for r in neg_rows)
    fp = sum(norm(r["prediction"]) == "yes" for r in neg_rows)

    acc = safe_div(tp + tn, len(pos_rows) + len(neg_rows))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return acc, precision, recall, specificity, f1

def delta_flip_metric(original_rows, masked_rows):
    original_map = {
        (r["image_id"], r["pathology"]): norm(r["prediction"])
        for r in original_rows
    }

    original_yes = 0
    yes_to_no = 0

    for r in masked_rows:
        key = (r["image_id"], r["pathology"])
        o = original_map.get(key, "missing")
        m = norm(r["prediction"])

        if o == "yes":
            original_yes += 1
            if m == "no":
                yes_to_no += 1

    return safe_div(yes_to_no, original_yes)

def paired_boot_delta(original_rows, masked_rows):
    # 按 key 做 paired bootstrap，保持 original/mask 对齐
    orig_map = {
        (r["image_id"], r["pathology"]): r
        for r in original_rows
    }
    mask_map = {
        (r["image_id"], r["pathology"]): r
        for r in masked_rows
    }

    keys = sorted(set(orig_map.keys()) & set(mask_map.keys()))
    vals = []

    for _ in range(N_BOOT):
        sampled_keys = [random.choice(keys) for _ in keys]
        o_sample = [orig_map[k] for k in sampled_keys]
        m_sample = [mask_map[k] for k in sampled_keys]
        vals.append(delta_flip_metric(o_sample, m_sample))

    return percentile(vals, 0.025), percentile(vals, 0.975)

def boot_classification(pos_rows, neg_rows):
    vals = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "f1": [],
    }

    for _ in range(N_BOOT):
        pos_s = [random.choice(pos_rows) for _ in pos_rows]
        neg_s = [random.choice(neg_rows) for _ in neg_rows]

        acc, prec, rec, spec, f1 = classification_metrics(pos_s, neg_s)

        vals["accuracy"].append(acc)
        vals["precision"].append(prec)
        vals["recall"].append(rec)
        vals["specificity"].append(spec)
        vals["f1"].append(f1)

    return {
        k: (percentile(v, 0.025), percentile(v, 0.975))
        for k, v in vals.items()
    }

def get_positive_original_path(model_key, dataset_prefix=""):
    if dataset_prefix:
        return RESULTS / f"{model_key}_{dataset_prefix}_original_results.jsonl"

    p1 = RESULTS / f"{model_key}_positive_original_results.jsonl"
    p2 = RESULTS / f"{model_key}_original_results.jsonl"
    return p1 if p1.exists() else p2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-prefix", default="", help="e.g. chestxdet10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(SEED)
    rows_out = []

    for model_key in MODEL_KEYS:
        print("processing", model_key)

        pos = load_jsonl(get_positive_original_path(model_key, args.dataset_prefix))

        if args.dataset_prefix:
            neg = load_jsonl(RESULTS / f"{model_key}_{args.dataset_prefix}_negative_original_results.jsonl")
            mp = load_jsonl(RESULTS / f"{model_key}_{args.dataset_prefix}_matched_patch_results.jsonl")
        else:
            neg = load_jsonl(RESULTS / f"{model_key}_negative_original_results.jsonl")
            mp = load_jsonl(RESULTS / f"{model_key}_matched_patch_results.jsonl")

        acc, prec, rec, spec, f1 = classification_metrics(pos, neg)
        cls_ci = boot_classification(pos, neg)

        mp_delta = delta_flip_metric(pos, mp)
        mp_lo, mp_hi = paired_boot_delta(pos, mp)

        sci = rec - mp_delta

        # SCI CI: 简单 bootstrap 近似，recall 和 delta 分开相减
        sci_lo = cls_ci["recall"][0] - mp_hi
        sci_hi = cls_ci["recall"][1] - mp_lo

        row = {
            "model_key": model_key,
            "model_name": MODEL_NAMES[model_key],

            "accuracy": acc,
            "accuracy_ci_low": cls_ci["accuracy"][0],
            "accuracy_ci_high": cls_ci["accuracy"][1],

            "precision": prec,
            "precision_ci_low": cls_ci["precision"][0],
            "precision_ci_high": cls_ci["precision"][1],

            "recall": rec,
            "recall_ci_low": cls_ci["recall"][0],
            "recall_ci_high": cls_ci["recall"][1],

            "specificity": spec,
            "specificity_ci_low": cls_ci["specificity"][0],
            "specificity_ci_high": cls_ci["specificity"][1],

            "f1": f1,
            "f1_ci_low": cls_ci["f1"][0],
            "f1_ci_high": cls_ci["f1"][1],

            "mp_delta_flip": mp_delta,
            "mp_delta_flip_ci_low": mp_lo,
            "mp_delta_flip_ci_high": mp_hi,

            "sci": sci,
            "sci_ci_low": sci_lo,
            "sci_ci_high": sci_hi,
        }

        rows_out.append(row)

    if args.out:
        out = Path(args.out)
    elif args.dataset_prefix:
        out = RESULTS / f"{args.dataset_prefix}_bootstrap_ci_summary.csv"
    else:
        out = RESULTS / "bootstrap_ci_summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print("saved ->", out)

    print("\nModel | Acc [CI] | Recall [CI] | MP ΔFlip [CI] | SCI [CI]")
    for r in rows_out:
        print(
            f'{r["model_name"]} | '
            f'{r["accuracy"]:.3f} [{r["accuracy_ci_low"]:.3f},{r["accuracy_ci_high"]:.3f}] | '
            f'{r["recall"]:.3f} [{r["recall_ci_low"]:.3f},{r["recall_ci_high"]:.3f}] | '
            f'{r["mp_delta_flip"]:.3f} [{r["mp_delta_flip_ci_low"]:.3f},{r["mp_delta_flip_ci_high"]:.3f}] | '
            f'{r["sci"]:.3f} [{r["sci_ci_low"]:.3f},{r["sci_ci_high"]:.3f}]'
        )

if __name__ == "__main__":
    main()
