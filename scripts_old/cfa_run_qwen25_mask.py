import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

sys.path.append("/home/217885@student.upm.edu.my/handoff_pkg/code")
from cfa_mask_operators import apply_bbox_mask

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PAIRS = Path("/home/217885@student.upm.edu.my/medcfa/data/medcfa_pairs_test.jsonl")

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", required=True, choices=["zero", "blur"])
    args = ap.parse_args()

    out = Path(f"/home/217885@student.upm.edu.my/medcfa/results/qwen25vl_{args.operator}_results.jsonl")
    tmp_dir = Path(f"/home/217885@student.upm.edu.my/medcfa/results/tmp_{args.operator}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL)

    pairs = [json.loads(line) for line in open(PAIRS)]
    print(f"running {len(pairs)} {args.operator} pairs...")

    with open(out, "w") as fout:
        for item in tqdm(pairs):
            img = Image.open(item["image_path"]).convert("RGB")
            masked = apply_bbox_mask(img, tuple(item["bbox"]), args.operator)

            safe_pathology = item["pathology"].replace(" ", "_")
            masked_path = tmp_dir / f'{item["image_id"]}_{safe_pathology}_{args.operator}.jpg'
            masked.save(masked_path)

            pred = ask(processor, model, str(masked_path), item["question"])

            item["condition"] = args.operator
            item["model"] = MODEL
            item["masked_image_path"] = str(masked_path)
            item["prediction"] = pred

            fout.write(json.dumps(item) + "\n")
            fout.flush()

    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
