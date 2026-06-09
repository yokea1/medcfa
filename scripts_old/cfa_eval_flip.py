import json
from collections import Counter, defaultdict

ORIG = "/home/217885@student.upm.edu.my/medcfa/results/qwen25vl_original_results.jsonl"
CF = "/home/217885@student.upm.edu.my/medcfa/results/qwen25vl_matched_patch_results.jsonl"

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
    orig[key(x)] = x

cf = {}
for line in open(CF):
    x = json.loads(line)
    cf[key(x)] = x

cnt = Counter()
by_path = defaultdict(Counter)

for k, o in orig.items():
    if k not in cf:
        continue

    op = norm(o["prediction"])
    cp = norm(cf[k]["prediction"])

    transition = f"{op}->{cp}"
    cnt[transition] += 1
    cnt["total"] += 1

    pathology = o["pathology"]
    by_path[pathology][transition] += 1
    by_path[pathology]["total"] += 1

orig_yes = cnt["yes->yes"] + cnt["yes->no"] + cnt["yes->other"]
flip = cnt["yes->no"]
flip_rate = flip / orig_yes if orig_yes else 0

print("=== Overall ===")
print("total =", cnt["total"])
print("yes->yes =", cnt["yes->yes"])
print("yes->no  =", cnt["yes->no"])
print("no->yes  =", cnt["no->yes"])
print("no->no   =", cnt["no->no"])
print("other transitions =", {k:v for k,v in cnt.items() if "other" in k})
print("original_yes =", orig_yes)
print("delta_flip =", flip_rate)

print("\n=== By pathology ===")
for p, c in sorted(by_path.items()):
    oy = c["yes->yes"] + c["yes->no"] + c["yes->other"]
    fr = c["yes->no"] / oy if oy else 0
    print(f"{p:30s} total={c['total']:4d} original_yes={oy:4d} yes->no={c['yes->no']:4d} delta_flip={fr:.4f}")
