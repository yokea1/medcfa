import csv
from pathlib import Path

patterns = [
    "*_ccf_summary.csv",
    "*_ccf_matched_patch_summary.csv",
    "*_ccf_zero_summary.csv",
    "*_ccf_blur_summary.csv",
]

files = []
for p in patterns:
    files.extend(Path("results").glob(p))

# 去重 + 排序
files = sorted(set(files))

rows = []
for f in files:
    with open(f, newline="") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            r["source_file"] = str(f)
            rows.append(r)

out = Path("results/ccf_summary_all_models_all_ops.csv")

if not rows:
    print("No CCF summary files found.")
else:
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved -> {out}")
    print(f"rows = {len(rows)}")
    print("operators =", sorted(set(r.get("operator", "") for r in rows)))
