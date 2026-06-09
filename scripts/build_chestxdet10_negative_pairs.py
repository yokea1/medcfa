import json
import random
from pathlib import Path

ROOT = Path("/home/217885@student.upm.edu.my/datasets/ChestX-Det10-Dataset")
ANN = ROOT / "test.json"
IMG_DIR = ROOT / "test_data/test_data"

OUT_ALL = Path("/home/217885@student.upm.edu.my/medcfa/data/chestxdet10_negative_all.jsonl")
OUT_SAMPLE = Path("/home/217885@student.upm.edu.my/medcfa/data/chestxdet10_negative_test.jsonl")

N_SAMPLE = 1476
SEED = 42

PATHOLOGIES = [
    "Atelectasis",
    "Calcification",
    "Consolidation",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Fracture",
    "Mass",
    "Nodule",
    "Pneumothorax",
]

random.seed(SEED)

data = json.load(open(ANN))
pairs = []

for item in data:
    file_name = item["file_name"]
    present = set(item.get("syms", []))

    img_path = IMG_DIR / file_name
    if not img_path.exists():
        continue

    image_id = Path(file_name).stem

    for pathology in PATHOLOGIES:
        if pathology in present:
            continue

        pairs.append({
            "image_id": f"{image_id}_{pathology.replace(' ', '_')}_neg",
            "image_path": str(img_path),
            "pathology": pathology,
            "answer": "no",
            "bbox": None,
            "question": f"Is there any sign of {pathology} in this image?",
            "dataset": "ChestX-Det10",
        })

random.shuffle(pairs)

OUT_ALL.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_ALL, "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

sample = pairs[:min(N_SAMPLE, len(pairs))]

with open(OUT_SAMPLE, "w") as f:
    for p in sample:
        f.write(json.dumps(p) + "\n")

print("all negative pairs =", len(pairs))
print("sampled negative pairs =", len(sample))
print("saved all ->", OUT_ALL)
print("saved sample ->", OUT_SAMPLE)
