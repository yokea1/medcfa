import json
import random
from pathlib import Path

import pandas as pd

DATA_ROOT = Path("/home/217885@student.upm.edu.my/medcfa/data/chexlocalize/raw/chexlocalize")
LABEL_CSV = DATA_ROOT / "CheXpert/test_labels.csv"
OUT_ALL = Path("/home/217885@student.upm.edu.my/medcfa/data/medcfa_pairs_negative_all.jsonl")
OUT_SAMPLE = Path("/home/217885@student.upm.edu.my/medcfa/data/medcfa_pairs_negative_test.jsonl")

N_SAMPLE = 911
SEED = 42

PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Lesion",
    "Pleural Effusion",
    "Pneumothorax",
]

random.seed(SEED)
df = pd.read_csv(LABEL_CSV)

pairs = []

for _, row in df.iterrows():
    img_rel = str(row["Path"])

    img_path = DATA_ROOT / "CheXpert" / img_rel
    if not img_path.exists():
        img_path = DATA_ROOT / img_rel

    if not img_path.exists():
        continue

    image_id = (
        str(img_path)
        .split("CheXpert/test/")[-1]
        .replace("/", "_")
        .replace(".jpg", "")
    )

    for pathology in PATHOLOGIES:
        if pathology not in row:
            continue

        val = row[pathology]

        # CheXpert label convention:
        # 1 = positive, 0 = negative, -1 = uncertain, blank = missing
        if pd.isna(val):
            continue

        try:
            val = float(val)
        except Exception:
            continue

        if val != 0:
            continue

        pairs.append({
            "image_id": image_id,
            "image_path": str(img_path),
            "pathology": pathology,
            "answer": "no",
            "bbox": None,
            "question": f"Is there any sign of {pathology} in this image?",
        })

random.shuffle(pairs)

OUT_ALL.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_ALL, "w") as f:
    for item in pairs:
        f.write(json.dumps(item) + "\n")

sample = pairs[:min(N_SAMPLE, len(pairs))]

with open(OUT_SAMPLE, "w") as f:
    for item in sample:
        f.write(json.dumps(item) + "\n")

print(f"all negative pairs = {len(pairs)}")
print(f"sampled negative pairs = {len(sample)}")
print(f"saved all -> {OUT_ALL}")
print(f"saved sample -> {OUT_SAMPLE}")
