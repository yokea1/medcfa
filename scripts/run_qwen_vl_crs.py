import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

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
    prompt = (
        "You are a radiologist reviewing a chest X-ray. "
        "Answer with one word: yes or no. "
        f"Question: {question}"
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def replace_patch_with_distractor(img, distractor, patch_box):
    x0, y0, x1, y1 = patch_box
    out = img.copy()

    # resize distractor to same image size
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
    ap.add_argument("--split", default="chestxdet10_crs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["model_name"]
    model_key = cfg["model_key"]
    model_type = cfg.get("model_type", "qwen_vl")

    out_path = RESULTS / f"{model_key}_{args.split}_grid{args.grid_size}_crs.jsonl"
    tmp_dir = RESULTS / f"tmp_{model_key}_{args.split}_grid{args.grid_size}_crs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("loading model:", model_name)

    if model_type == "qwen25_vl":
        model_cls = Qwen2_5_VLForConditionalGeneration
    elif model_type == "qwen3_vl":
        model_cls = Qwen3VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
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

            # CRS only meaningful if original says yes, but still record skipped cases
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
