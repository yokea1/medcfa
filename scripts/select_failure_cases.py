import csv
import json
from pathlib import Path

RESULTS = Path("results")
OUT = RESULTS / "failure_case_candidates.csv"

CASES = [
    {
        "case_type": "high_sci_shortcut",
        "model_key": "internvl3",
        "model_name": "InternVL3",
        "target_transition": ("yes", "yes"),
        "description": "High-performance model remains yes after matched-patch masking",
    },
    {
        "case_type": "grounding_sensitive_huatuo",
        "model_key": "huatuogpt_vision",
        "model_name": "HuatuoGPT-Vision",
        "target_transition": ("yes", "no"),
        "description": "Medical model flips from yes to no after matched-patch masking",
    },
    {
        "case_type": "grounding_sensitive_medgemma",
        "model_key": "medgemma",
        "model_name": "MedGemma",
        "target_transition": ("yes", "no"),
        "description": "Medical model flips from yes to no after matched-patch masking",
    },
    {
        "case_type": "low_grounding_llava_med",
        "model_key": "llava_med",
        "model_name": "LLaVA-Med",
        "target_transition": ("yes", "yes"),
        "description": "Low-grounding model keeps yes after matched-patch masking",
    },
]

PREFERRED_PATHOLOGIES = [
    "Cardiomegaly",
    "Enlarged Cardiomediastinum",
    "Atelectasis",
    "Edema",
    "Pleural Effusion",
    "Consolidation",
    "Lung Lesion",
    "Pneumothorax",
]

def norm(x):
    x = str(x).strip().lower()
    if x.startswith("yes"):
        return "yes"
    if x.startswith("no"):
        return "no"
    return "other"

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def get_positive_original_path(model_key):
    p1 = RESULTS / f"{model_key}_positive_original_results.jsonl"
    p2 = RESULTS / f"{model_key}_original_results.jsonl"
    return p1 if p1.exists() else p2

def index_rows(rows):
    return {
        (r["image_id"], r["pathology"]): r
        for r in rows
    }

def pathology_rank(pathology):
    try:
        return PREFERRED_PATHOLOGIES.index(pathology)
    except ValueError:
        return 999

def select_cases():
    rows_out = []

    for spec in CASES:
        model_key = spec["model_key"]
        original = index_rows(load_jsonl(get_positive_original_path(model_key)))
        matched = index_rows(load_jsonl(RESULTS / f"{model_key}_matched_patch_results.jsonl"))

        target_o, target_m = spec["target_transition"]

        candidates = []

        for key, o_row in original.items():
            if key not in matched:
                continue

            m_row = matched[key]

            o_pred = norm(o_row["prediction"])
            m_pred = norm(m_row["prediction"])

            if o_pred != target_o or m_pred != target_m:
                continue

            candidates.append({
                "case_type": spec["case_type"],
                "description": spec["description"],
                "model_key": model_key,
                "model_name": spec["model_name"],
                "image_id": o_row["image_id"],
                "pathology": o_row["pathology"],
                "answer": o_row["answer"],
                "bbox": o_row["bbox"],
                "image_path": o_row["image_path"],
                "matched_patch_image_path": m_row.get("masked_image_path", ""),
                "original_prediction": o_row["prediction"],
                "matched_patch_prediction": m_row["prediction"],
                "transition": f"{o_pred}->{m_pred}",
            })

        candidates.sort(key=lambda x: (pathology_rank(x["pathology"]), x["image_id"]))

        # 每类取前 10 个，方便人工挑图
        rows_out.extend(candidates[:10])

    return rows_out

def main():
    rows = select_cases()

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("saved ->", OUT)
    print(f"total candidates = {len(rows)}")

    print("\nPreview:")
    for r in rows[:20]:
        print(
            f'{r["case_type"]:<30} '
            f'{r["model_name"]:<20} '
            f'{r["pathology"]:<30} '
            f'{r["transition"]:<8} '
            f'{r["image_id"]}'
        )

if __name__ == "__main__":
    main()
