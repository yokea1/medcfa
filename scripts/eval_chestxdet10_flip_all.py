import csv
import subprocess
from pathlib import Path

RESULTS = Path("results")
OUT = RESULTS / "chestxdet10_flip_all_summary.csv"

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

def parse_eval_output(text):
    rows = []
    lines = [x.strip() for x in text.strip().splitlines() if x.strip()]
    header = lines[0].split(",")

    for line in lines[1:]:
        values = line.split(",")
        rows.append(dict(zip(header, values)))

    return rows

def main():
    all_rows = []

    for model_key in MODEL_KEYS:
        full_key = f"{model_key}_chestxdet10"

        cmd = [
            "python",
            "scripts/eval_flip.py",
            "--model-key",
            full_key,
        ]

        print("running:", " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        rows = parse_eval_output(result.stdout)

        for r in rows:
            r["model_key"] = model_key
            r["model_name"] = MODEL_NAMES[model_key]
            r["dataset"] = "ChestX-Det10"
            all_rows.append(r)

    fieldnames = [
        "dataset",
        "model_key",
        "model_name",
        "operator",
        "total",
        "original_yes",
        "yes_to_no",
        "yes_to_yes",
        "no_to_yes",
        "no_to_no",
        "other_transitions",
        "delta_flip",
    ]

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("saved ->", OUT)

    print("\nSummary:")
    for r in all_rows:
        if r["operator"] == "matched_patch":
            print(
                f'{r["model_name"]}: '
                f'MP ΔFlip={float(r["delta_flip"]):.4f}, '
                f'original_yes={r["original_yes"]}/{r["total"]}'
            )

if __name__ == "__main__":
    main()
