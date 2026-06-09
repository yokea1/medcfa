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
from cfa_mask_operators import apply_bbox_mask, load_distractor

PAIRS = Path("/home/217885@student.upm.edu.my/medcfa/data/medcfa_pairs_test.jsonl")
RESULTS = Path("/home/217885@student.upm.edu.my/medcfa/results")
DISTRACTOR = "/home/217885@student.upm.edu.my/medcfa/data/chexlocalize/raw/chexlocalize/CheXpert/test/patient65079/study1/view1_frontal.jpg"

MODEL = "/home/217885@student.upm.edu.my/medcfa/cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2"
MODEL_KEY = "llava_med"

def parse_answer(raw):
    x = raw.lower().strip()
    if "answer: yes" in x or x.startswith("yes"):
        return "yes"
    if "answer: no" in x or x.startswith("no"):
        return "no"
    if " yes" in x:
        return "yes"
    if " no" in x:
        return "no"
    return raw.strip()

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
    return parse_answer(raw), raw

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
    ap.add_argument("--condition", required=True, choices=["original", "matched_patch", "zero", "blur"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--split", default="positive")
    args = ap.parse_args()

    out = RESULTS / f"{MODEL_KEY}_{args.split}_{args.condition}_results.jsonl"
    tmp_dir = RESULTS / f"tmp_{MODEL_KEY}_{args.condition}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("loading official LLaVA-Med...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL,
        model_base=None,
        model_name="llava_mistral",
        device_map="auto",
    )
    model.eval()

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
            pred, raw = ask(tokenizer, model, image_processor, image_path, item["question"])

            item["condition"] = args.condition
            item["model"] = "microsoft/llava-med-v1.5-mistral-7b"
            item["prediction"] = pred
            item["raw_prediction"] = raw

            if args.condition != "original":
                item["masked_image_path"] = image_path

            fout.write(json.dumps(item) + "\n")
            fout.flush()

    print("saved ->", out)

if __name__ == "__main__":
    main()
