import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

PAIRS = Path("/home/217885@student.upm.edu.my/medcfa/data/medcfa_pairs_test.jsonl")
RESULTS = Path("/home/217885@student.upm.edu.my/medcfa/results")

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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen25vl.yaml")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["model_name"]

    out = RESULTS / "audit_qwen25_raw50.jsonl"

    print("loading model:", model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    pairs = [json.loads(line) for line in open(PAIRS)]
    pairs = pairs[:args.limit]

    print(f"running {len(pairs)} qwen25 audit samples")

    with open(out, "w") as fout:
        for item in tqdm(pairs):
            raw_pred = ask(processor, model, item["image_path"], item["question"])

            item["condition"] = "original"
            item["model"] = model_name
            item["raw_prediction"] = raw_pred
            item["prediction"] = raw_pred

            fout.write(json.dumps(item) + "\n")
            fout.flush()

    print("saved ->", out)

if __name__ == "__main__":
    main()
