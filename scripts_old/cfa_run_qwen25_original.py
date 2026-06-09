import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PAIRS = Path("/home/217885@student.upm.edu.my/medcfa/data/medcfa_pairs_test.jsonl")
OUT = Path("/home/217885@student.upm.edu.my/medcfa/results/qwen25vl_original_results.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL)

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

print(f"running {len(pairs)} original pairs...")

with open(OUT, "w") as fout:
    for item in tqdm(pairs):
        pred = ask(item["image_path"], item["question"])
        item["condition"] = "original"
        item["model"] = MODEL
        item["prediction"] = pred
        fout.write(json.dumps(item) + "\n")
        fout.flush()

print(f"saved -> {OUT}")
