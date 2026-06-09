import csv
import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path("results")

MODELS = [
    ("qwen25vl", "Qwen2.5-VL"),
    ("qwen3vl", "Qwen3-VL"),
    ("huatuogpt_vision", "HuatuoGPT-Vision"),
    ("medgemma", "MedGemma"),
    ("internvl3", "InternVL3"),
    ("llava_med", "LLaVA-Med"),
]

OUT = RESULTS / "chestxdet10_per_pathology_breakdown.csv"

rows = []

for model_key, model_name in MODELS:

    orig_path = RESULTS / f"{model_key}_chestxdet10_original_results.jsonl"
    mp_path = RESULTS / f"{model_key}_chestxdet10_matched_patch_results.jsonl"

    orig = [json.loads(x) for x in open(orig_path)]
    mp = [json.loads(x) for x in open(mp_path)]

    by_pathology = defaultdict(list)

    for o, m in zip(orig, mp):
        by_pathology[o["pathology"]].append((o, m))

    for pathology, pairs in by_pathology.items():

        total = len(pairs)

        orig_yes = 0
        yes_to_no = 0

        for o, m in pairs:

            op = o["prediction"].lower()
            mpred = m["prediction"].lower()

            o_yes = op.startswith("y")
            m_yes = mpred.startswith("y")

            if o_yes:
                orig_yes += 1

            if o_yes and (not m_yes):
                yes_to_no += 1

        delta_flip = yes_to_no / orig_yes if orig_yes else 0

        rows.append({
            "model_key": model_key,
            "model_name": model_name,
            "pathology": pathology,
            "total": total,
            "original_yes": orig_yes,
            "delta_flip": delta_flip,
        })

with open(OUT, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model_key",
            "model_name",
            "pathology",
            "total",
            "original_yes",
            "delta_flip",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

print("saved ->", OUT)

print("\nTop vulnerable pathologies")

for model_key, model_name in MODELS:

    sub = [r for r in rows if r["model_key"] == model_key]

    sub.sort(key=lambda x: x["delta_flip"], reverse=True)

    print("\n", model_name)

    for r in sub[:3]:
        print(
            f'{r["pathology"]:<15} '
            f'n={r["total"]:<4} '
            f'orig_yes={r["original_yes"]:<4} '
            f'MP_Delta={r["delta_flip"]:.4f}'
        )
