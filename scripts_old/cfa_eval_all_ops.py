import json
from collections import Counter

ORIG = "results/qwen25vl_original_results.jsonl"
OPS = {
    "matched_patch": "results/qwen25vl_matched_patch_results.jsonl",
    "zero": "results/qwen25vl_zero_results.jsonl",
    "blur": "results/qwen25vl_blur_results.jsonl",
}

def norm(x):
    x = x.strip().lower()
    if x.startswith("yes"):
        return "yes"
    if x.startswith("no"):
        return "no"
    return "other"

def key(x):
    return (x["image_id"], x["pathology"])

orig = {}
for line in open(ORIG):
    x = json.loads(line)
    orig[key(x)] = norm(x["prediction"])

print("operator,total,original_yes,yes_to_no,yes_to_yes,no_to_yes,no_to_no,delta_flip")

for op, path in OPS.items():
    cf = {}
    for line in open(path):
        x = json.loads(line)
        cf[key(x)] = norm(x["prediction"])

    c = Counter()
    for k, opred in orig.items():
        cpred = cf[k]
        c[f"{opred}->{cpred}"] += 1
        c["total"] += 1

    original_yes = c["yes->yes"] + c["yes->no"] + c["yes->other"]
    yes_to_no = c["yes->no"]
    delta_flip = yes_to_no / original_yes if original_yes else 0

    print(
        f"{op},{c['total']},{original_yes},{yes_to_no},"
        f"{c['yes->yes']},{c['no->yes']},{c['no->no']},{delta_flip:.4f}"
    )
