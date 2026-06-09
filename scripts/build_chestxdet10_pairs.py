import json
from pathlib import Path

ROOT = Path("/home/217885@student.upm.edu.my/datasets/ChestX-Det10-Dataset")
ANN = ROOT / "test.json"
IMG_DIR = ROOT / "test_data/test_data"

OUT = Path("/home/217885@student.upm.edu.my/medcfa/data/chestxdet10_pairs_test.jsonl")

data = json.load(open(ANN))

pairs = []

for item in data:
    file_name = item["file_name"]
    syms = item.get("syms", [])
    boxes = item.get("boxes", [])

    img_path = IMG_DIR / file_name
    if not img_path.exists():
        continue

    if not syms or not boxes:
        continue

    # syms 和 boxes 一一对应
    for idx, (pathology, bbox) in enumerate(zip(syms, boxes)):
        image_id = Path(file_name).stem

        pairs.append({
            "image_id": f"{image_id}_{idx}",
            "image_path": str(img_path),
            "pathology": pathology,
            "answer": "yes",
            "bbox": bbox,
            "question": f"Is there any sign of {pathology} in this image?",
            "dataset": "ChestX-Det10",
        })

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

print("saved ->", OUT)
print("pairs =", len(pairs))
