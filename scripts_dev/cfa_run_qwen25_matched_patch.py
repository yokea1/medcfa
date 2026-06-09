import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import sys
sys.path.append("/home/217885@student.upm.edu.my/handoff_pkg/code")
from cfa_mask_operators import apply_bbox_mask, load_distractor

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PAIRS = Path("/home/217885@student.upm.edu.my/medcfa/data/medcfa_pairs_test.jsonl")
OUT = Path("/home/217885@student.upm.edu.my/medcfa/results/qwen25vl_matched_patch_results.jsonl")
TMP_DIR = Path("/home/217885@student.upm.edu.my/medcfa/results/tmp_matched_patch")
DISTRACTOR = "/home/217885@student.upm.edu.my/medcfa/data/chexlocalize/raw/chexlocalize/CheXpert/test/patient65079/study1/view1_frontal.jpg"

OUT.parent.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

print("loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL)
distractor = load_distractor(DISTRACTOR)

def make_masked_image(item):
    img = Image.open(item["image_path"]).convert("RGB")
    masked = apply_bbox_mask(
        img,
        tuple(item["bbox"]),
        "matched_patch",
        distractor_image=distractor,
    )
    out_path = TMP_DIR / f'{item["image_id"]}_{item["pathology"].replace(" ", "_")}_matched_patch.jpg'
    masked.save(out_path)
    return str(out_path)

def ask(image_path, question):
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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    ans = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return ans

with open(PAIRS) as f:
    pairs = [json.loads(line) for line in f]

print(f"running {len(pairs)} matched_patch pairs...")

with open(OUT, "w") as fout:
    for item in tqdm(pairs):
        masked_path = make_masked_image(item)
        pred = ask(masked_path, item["question"])

        item["condition"] = "matched_patch"
        item["model"] = MODEL
        item["masked_image_path"] = masked_path
        item["prediction"] = pred

        fout.write(json.dumps(item) + "\n")
        fout.flush()

print(f"saved -> {OUT}")
