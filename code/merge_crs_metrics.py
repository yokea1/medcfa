import csv
from pathlib import Path

files = sorted(Path("results").glob("*_chestxdet10_crs_100_metrics.csv"))

rows = []
for f in files:
    model_key = f.name.replace("_chestxdet10_crs_100_metrics.csv", "")
    with open(f, newline="") as fp:
        reader = csv.DictReader(fp)
        data = list(reader)

    total = len(data)
    with_flip = sum(int(r.get("num_flips", 0)) > 0 for r in data)
    hit1 = sum(int(r.get("hit1", 0)) for r in data)
    intersect1 = sum(int(r.get("intersect1", 0)) for r in data)

    # skipped rows are not in metric csv rows if no patches; infer from printed total=100
    analyzable = total
    skipped = 100 - analyzable

    rows.append({
        "model_key": model_key,
        "total": 100,
        "skipped": skipped,
        "analyzable": analyzable,
        "with_flip": with_flip,
        "flip_case_rate": with_flip / analyzable if analyzable else 0,
        "hit1_rate": hit1 / with_flip if with_flip else 0,
        "intersect1_rate": intersect1 / with_flip if with_flip else 0,
        "source_file": str(f),
    })

out = Path("results/crs_metrics_all_models.csv")
with open(out, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"saved -> {out}")
print(f"rows = {len(rows)}")

for r in rows:
    print(
        r["model_key"],
        "analyzable=", r["analyzable"],
        "flip=", f'{r["flip_case_rate"]:.4f}',
        "hit1=", f'{r["hit1_rate"]:.4f}',
        "intersect=", f'{r["intersect1_rate"]:.4f}',
    )
