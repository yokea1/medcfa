import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

MODEL = "google/medgemma-4b-it"
IMAGE = "/home/217885@student.upm.edu.my/medcfa/data/chexlocalize/raw/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"

print("loading MedGemma...")
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

image = Image.open(IMAGE).convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "You are a radiologist reviewing a chest X-ray. Answer with exactly one word: Yes or No. Question: Is there any sign of Cardiomegaly in this image?"},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

print("running inference...")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=8, do_sample=False)

new_tokens = output[0][inputs["input_ids"].shape[-1]:]
answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
print("answer:", repr(answer))
