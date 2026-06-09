import json
import csv
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

DATA = Path("/home/217885@student.upm.edu.my/medcfa/data")
OUT = Path("results/bbox_size_diagnostic_summary.csv")
OUT_PATH = Path("results/bbox_size_per_pathology_summary.csv")

DATASETS = {
    "chexlocalize": DATA / "medcfa_pairs_test.jsonl",
    "chestxdet10": DATA / "chestxdet10_pairs_test.jsonl",
}

rows = []
path_rows = []

for ds, pair_file in DATASETS.items():
    vals = []
    per_path = defaultdict(list)

    for line in open(pair_file):
        r = json.loads(line)
        if not r.get("bbox"):
            continue

        img = Image.open(r["image_path"])
        w, h = img.size
        x1, y1, x2, y2 = r["bbox"]

        frac = ((x2 - x1) * (y2 - y1)) / (w * h)

        vals.append(frac)
        per_path[r["pathology"]].append(frac)

    arr = np.array(vals)

    rows.append({
        "dataset": ds,
        "n": len(vals),
        "mean": arr.mean(),
        "median": np.median(arr),
        "p05": np.percentile(arr, 5),
        "p25": np.percentile(arr, 25),
        "p75": np.percentile(arr, 75),
        "p95": np.percentile(arr, 95),
    })

    for path, xs in per_path.items():
        a = np.array(xs)
        path_rows.append({
            "dataset": ds,
            "pathology": path,
            "n": len(xs),
            "mean": a.mean(),
            "median": np.median(a),
            "p25": np.percentile(a, 25),
            "p75": np.percentile(a, 75),
        })

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

with open(OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=path_rows[0].keys())
    writer.writeheader()
    writer.writerows(path_rows)

print("saved ->", OUT)
print("saved ->", OUT_PATH)

print("\nDataset bbox area fraction summary:")
for r in rows:
    print(
        f'{r["dataset"]}: n={r["n"]}, '
        f'mean={r["mean"]:.4f}, median={r["median"]:.4f}, '
        f'p05={r["p05"]:.4f}, p95={r["p95"]:.4f}'
    )
