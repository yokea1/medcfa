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
from cfa_mask_operators import apply_bbox_mask, load_distractor

PAIRS = Path("/home/217885@student.upm.edu.my/medcfa/data/medcfa_pairs_test.jsonl")
RESULTS = Path("/home/217885@student.upm.edu.my/medcfa/results")
DISTRACTOR = "/home/217885@student.upm.edu.my/medcfa/data/chexlocalize/raw/chexlocalize/CheXpert/test/patient65079/study1/view1_frontal.jpg"

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
    ap.add_argument("--config", required=True)
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

    model_type = cfg.get("model_type", "qwen_vl")

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
