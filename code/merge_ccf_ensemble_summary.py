import csv
from pathlib import Path

files = sorted(Path("results").glob("*_ccf_ensemble_summary.csv"))
rows = []

for f in files:
    with open(f, newline="") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            r["source_file"] = str(f)
            rows.append(r)

out = Path("results/ccf_ensemble_summary_all_models.csv")

if rows:
    with open(out, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

print(f"saved -> {out}")
print(f"rows = {len(rows)}")
