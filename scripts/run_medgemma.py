import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

import sys
sys.path.append("/home/217885@student.upm.edu.my/handoff_pkg/code")
from cfa_mask_operators import apply_bbox_mask, load_distractor

PAIRS = Path("/home/217885@student.upm.edu.my/medcfa/data/medcfa_pairs_test.jsonl")
RESULTS = Path("/home/217885@student.upm.edu.my/medcfa/results")
DISTRACTOR = "/home/217885@student.upm.edu.my/medcfa/data/chexlocalize/raw/chexlocalize/CheXpert/test/patient65079/study1/view1_frontal.jpg"

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

def make_image(item, condition, tmp_dir, distractor=None):
    if condition == "original":
        return item["image_path"]

    img = Image.open(item["image_path"]).convert("RGB")

    if condition == "matched_patch":
        masked = apply_bbox_mask(
            img,
            tuple(item["bbox"]),
            "matched_patch",
            distractor_image=distractor,
        )
    else:
        masked = apply_bbox_mask(img, tuple(item["bbox"]), condition)

    safe_pathology = item["pathology"].replace(" ", "_")
    out_path = tmp_dir / f'{item["image_id"]}_{safe_pathology}_{condition}.jpg'
    masked.save(out_path)
    return str(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/medgemma.yaml")
    ap.add_argument("--condition", required=True, choices=["original", "matched_patch", "zero", "blur"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--split", default="positive")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["model_name"]
    model_key = cfg["model_key"]

    out = RESULTS / f"{model_key}_{args.split}_{args.condition}_results.jsonl"
    tmp_dir = RESULTS / f"tmp_{model_key}_{args.condition}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("loading model:", model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    distractor = None
    if args.condition == "matched_patch":
        distractor = load_distractor(DISTRACTOR)

    pairs = [json.loads(line) for line in open(args.pairs)]
    if args.limit is not None:
        pairs = pairs[:args.limit]

    print(f"running {len(pairs)} pairs, condition={args.condition}")

    with open(out, "w") as fout:
        for item in tqdm(pairs):
            image_path = make_image(item, args.condition, tmp_dir, distractor)
            pred = ask(processor, model, image_path, item["question"])

            item["condition"] = args.condition
            item["model"] = model_name
            item["prediction"] = pred

            if args.condition != "original":
                item["masked_image_path"] = image_path

            fout.write(json.dumps(item) + "\n")
            fout.flush()

    print("saved ->", out)

if __name__ == "__main__":
    main()
