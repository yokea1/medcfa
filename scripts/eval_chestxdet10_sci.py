import csv
from pathlib import Path

RESULTS = Path("results")

CLS = RESULTS / "chestxdet10_classification_summary_with_negative.csv"
FLIP = RESULTS / "chestxdet10_flip_all_summary.csv"

OUT = RESULTS / "chestxdet10_sci_summary.csv"

flip_map = {}

with open(FLIP) as f:
    for r in csv.DictReader(f):
        if r["operator"] == "matched_patch":
            flip_map[r["model_key"]] = float(r["delta_flip"])

rows = []

with open(CLS) as f:
    for r in csv.DictReader(f):

        model_key = r["model_key"]

        recall = float(r["recall"])

        mp_flip = flip_map[model_key]

        sci = recall - mp_flip

        rows.append({
            "model_key": model_key,
            "model_name": r["model_name"],
            "recall": recall,
            "matched_patch_delta_flip": mp_flip,
            "sci": sci,
        })

rows.sort(key=lambda x: x["sci"], reverse=True)

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model_key",
            "model_name",
            "recall",
            "matched_patch_delta_flip",
            "sci",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

print("saved ->", OUT)

print()
print("Model | Recall | MP ΔFlip | SCI")

for r in rows:
    print(
        f'{r["model_name"]} | '
        f'{r["recall"]:.4f} | '
        f'{r["matched_patch_delta_flip"]:.4f} | '
        f'{r["sci"]:.4f}'
    )
