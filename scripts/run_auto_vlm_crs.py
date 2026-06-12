import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.append("/home/217885@student.upm.edu.my/handoff_pkg/code")
from cfa_mask_operators import load_distractor

RESULTS = Path("/home/217885@student.upm.edu.my/medcfa/results")
DISTRACTOR = "/home/217885@student.upm.edu.my/medcfa/data/chexlocalize/raw/chexlocalize/CheXpert/test/patient65079/study1/view1_frontal.jpg"


def norm_pred(x):
    x = str(x).strip().lower()
    for ch in ".!?,;:":
        x = x.replace(ch, "")
    first = x.split()[0] if x.split() else ""
    if first.startswith("yes"):
        return "yes"
    if first.startswith("no"):
        return "no"
    return "other"


def ask(processor, model, image_path, question):
    image = Image.open(image_path).convert("RGB")

    prompt = (
        "You are a radiologist reviewing a chest X-ray. "
        "Answer with exactly one word: Yes or No. "
        f"Question: {question}"
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def replace_patch_with_distractor(img, distractor, patch_box):
    x0, y0, x1, y1 = patch_box
    out = img.copy()
    dis = distractor.resize(img.size)
    patch = dis.crop((x0, y0, x1, y1))
    out.paste(patch, (x0, y0))
    return out


def make_grid_boxes(width, height, grid_size):
    boxes = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            x0 = int(round(gx * width / grid_size))
            x1 = int(round((gx + 1) * width / grid_size))
            y0 = int(round(gy * height / grid_size))
            y1 = int(round((gy + 1) * height / grid_size))
            boxes.append((gx, gy, [x0, y0, x1, y1]))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs", default="data/crs_chestxdet10_100_positive.jsonl")
    ap.add_argument("--grid-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="chestxdet10_crs_100")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["model_name"]
    model_key = cfg["model_key"]

    out_path = RESULTS / f"{model_key}_{args.split}_grid{args.grid_size}_crs.jsonl"
    tmp_dir = RESULTS / f"tmp_{model_key}_{args.split}_grid{args.grid_size}_crs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("loading model:", model_name)

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    distractor = load_distractor(DISTRACTOR).convert("RGB")

    pairs = [json.loads(line) for line in open(args.pairs) if line.strip()]
    if args.limit is not None:
        pairs = pairs[:args.limit]

    print(f"running CRS pairs={len(pairs)}, grid={args.grid_size}")
    print("saved ->", out_path)

    with open(out_path, "w") as fout:
        for item in tqdm(pairs):
            img = Image.open(item["image_path"]).convert("RGB")
            w, h = img.size

            original_raw = ask(processor, model, item["image_path"], item["question"])
            original_pred = norm_pred(original_raw)

            result = {
                "image_id": item.get("image_id"),
                "image_path": item.get("image_path"),
                "pathology": item.get("pathology"),
                "dataset": item.get("dataset", "ChestX-Det10"),
                "answer": item.get("answer"),
                "bbox": item.get("bbox"),
                "question": item.get("question"),
                "model": model_name,
                "model_key": model_key,
                "grid_size": args.grid_size,
                "original_raw_prediction": original_raw,
                "original_prediction": original_pred,
                "patches": [],
            }

            if original_pred != "yes":
                result["skipped"] = True
                result["skip_reason"] = "original_not_yes"
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            result["skipped"] = False

            for gx, gy, box in make_grid_boxes(w, h, args.grid_size):
                patched = replace_patch_with_distractor(img, distractor, tuple(box))

                safe_pathology = item["pathology"].replace(" ", "_")
                patch_path = tmp_dir / f'{item["image_id"]}_{safe_pathology}_g{gx}_{gy}.jpg'
                patched.save(patch_path)

                raw = ask(processor, model, str(patch_path), item["question"])
                pred = norm_pred(raw)
                flip = int(original_pred == "yes" and pred == "no")

                result["patches"].append({
                    "gx": gx,
                    "gy": gy,
                    "box": box,
                    "raw_prediction": raw,
                    "prediction": pred,
                    "flip": flip,
                    "patched_image_path": str(patch_path),
                })

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    print("done ->", out_path)


if __name__ == "__main__":
    main()
