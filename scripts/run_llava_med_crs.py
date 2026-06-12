import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/217885@student.upm.edu.my/LLaVA-Med")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

sys.path.append("/home/217885@student.upm.edu.my/handoff_pkg/code")
from cfa_mask_operators import load_distractor

RESULTS = Path("/home/217885@student.upm.edu.my/medcfa/results")
DISTRACTOR = "/home/217885@student.upm.edu.my/medcfa/data/chexlocalize/raw/chexlocalize/CheXpert/test/patient65079/study1/view1_frontal.jpg"

MODEL = "/home/217885@student.upm.edu.my/medcfa/cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2"
MODEL_KEY = "llava_med"


def norm_pred(x):
    x = str(x).strip().lower()
    for ch in ".!?,;:":
        x = x.replace(ch, "")
    first = x.split()[0] if x.split() else ""
    if first.startswith("yes") or "answer yes" in x or "answer: yes" in x:
        return "yes"
    if first.startswith("no") or "answer no" in x or "answer: no" in x:
        return "no"
    if " yes" in x:
        return "yes"
    if " no" in x:
        return "no"
    return "other"


def ask(tokenizer, model, image_processor, image_path, question):
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.to(dtype=torch.float16, device=model.device)

    prompt_text = (
        DEFAULT_IMAGE_TOKEN + "\n"
        "You are a radiologist reviewing a chest X-ray. "
        "Answer with exactly one word: Yes or No. "
        f"Question: {question}"
    )

    conv = conv_templates["plain"].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0),
            do_sample=False,
            max_new_tokens=32,
            use_cache=True,
        )

    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return raw.strip(), norm_pred(raw)


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
    ap.add_argument("--pairs", default="data/crs_chestxdet10_100_positive.jsonl")
    ap.add_argument("--grid-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="chestxdet10_crs_100")
    args = ap.parse_args()

    out_path = RESULTS / f"{MODEL_KEY}_{args.split}_grid{args.grid_size}_crs.jsonl"
    tmp_dir = RESULTS / f"tmp_{MODEL_KEY}_{args.split}_grid{args.grid_size}_crs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("loading official LLaVA-Med...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL,
        model_base=None,
        model_name="llava_mistral",
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

            original_raw, original_pred = ask(
                tokenizer, model, image_processor, item["image_path"], item["question"]
            )

            result = {
                "image_id": item.get("image_id"),
                "image_path": item.get("image_path"),
                "pathology": item.get("pathology"),
                "dataset": item.get("dataset", "ChestX-Det10"),
                "answer": item.get("answer"),
                "bbox": item.get("bbox"),
                "question": item.get("question"),
                "model": "microsoft/llava-med-v1.5-mistral-7b",
                "model_key": MODEL_KEY,
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

                raw, pred = ask(tokenizer, model, image_processor, str(patch_path), item["question"])
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
