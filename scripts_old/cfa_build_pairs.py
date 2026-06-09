import argparse
import json
from pathlib import Path

import pandas as pd


PATHOLOGIES = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
]


def path_to_image_id(path: str) -> str:
    p = Path(path)
    patient = p.parts[1]
    study = p.parts[2]
    view = p.stem
    return f"{patient}_{study}_{view}"


def polygon_to_bbox(polygons):
    xs, ys = [], []
    for poly in polygons:
        for x, y in poly:
            xs.append(float(x))
            ys.append(float(y))
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    labels_path = data_root / "CheXpert/test_labels.csv"
    ann_path = data_root / "CheXlocalize/gt_annotations_test.json"
    img_root = data_root / "CheXpert"

    labels = pd.read_csv(labels_path)
    anns = json.load(open(ann_path))

    pairs = []

    for _, row in labels.iterrows():
        rel_path = row["Path"]
        image_id = path_to_image_id(rel_path)
        image_path = img_root / rel_path

        if not image_path.exists():
            continue

        ann = anns.get(image_id, {})

        for pathology in PATHOLOGIES:
            val = row.get(pathology, 0)

            if pd.isna(val) or float(val) != 1.0:
                continue

            if pathology not in ann:
                continue

            bbox = polygon_to_bbox(ann[pathology])
            q = f"Is there any sign of {pathology} in this image?"

            pairs.append({
                "image_id": image_id,
                "image_path": str(image_path),
                "pathology": pathology,
                "answer": "yes",
                "bbox": bbox,
                "question": q,
            })

            if args.limit and len(pairs) >= args.limit:
                break

        if args.limit and len(pairs) >= args.limit:
            break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        for item in pairs:
            f.write(json.dumps(item) + "\n")

    print(f"saved {len(pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()
