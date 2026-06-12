import json
import random
from collections import defaultdict
from pathlib import Path

INPUT = "data/chestxdet10_pairs_test.jsonl"
OUTPUT = "data/crs_chestxdet10_100_positive.jsonl"
SEED = 42
N = 100

random.seed(SEED)

rows = []
with open(INPUT) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        ans = str(r.get("answer", "")).lower()
        if ans.startswith("yes") and r.get("bbox") is not None:
            rows.append(r)

by_path = defaultdict(list)
for r in rows:
    by_path[r["pathology"]].append(r)

selected = []

# 尽量每个病种先抽 10 个
for path, items in sorted(by_path.items()):
    random.shuffle(items)
    selected.extend(items[:10])

# 如果超过 100，截断
selected = selected[:N]

# 如果不够 100，再补
if len(selected) < N:
    used = set((r["image_id"], r["pathology"]) for r in selected)
    rest = [r for r in rows if (r["image_id"], r["pathology"]) not in used]
    random.shuffle(rest)
    selected.extend(rest[:N - len(selected)])

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    for r in selected:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"input positives = {len(rows)}")
print(f"saved -> {OUTPUT}")
print(f"selected = {len(selected)}")

cnt = defaultdict(int)
for r in selected:
    cnt[r["pathology"]] += 1

print("pathology distribution:")
for k, v in sorted(cnt.items()):
    print(k, v)
