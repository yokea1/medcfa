import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE = "/home/217885@student.upm.edu.my/medcfa/data/chexlocalize/raw/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"
QUESTION = "You are a radiologist reviewing a chest X-ray. Answer with one word: yes or no. Question: Is there any sign of Cardiomegaly in this image?"

print("loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE},
            {"type": "text", "text": QUESTION},
        ],
    }
]

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

print("running inference...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("answer:", output_text[0])
