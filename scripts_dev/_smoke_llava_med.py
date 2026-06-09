import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    LlavaProcessor,
    LlavaForConditionalGeneration,
)

MODEL = "microsoft/llava-med-v1.5-mistral-7b"
IMAGE = "/home/217885@student.upm.edu.my/medcfa/data/chexlocalize/raw/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"

print("loading LLaVA-Med...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

image = Image.open(IMAGE).convert("RGB")

prompt = (
    "USER: <image>\n"
    "You are a radiologist reviewing a chest X-ray. "
    "Answer with exactly one word: Yes or No. "
    "Question: Is there any sign of Cardiomegaly in this image?\n"
    "ASSISTANT:"
)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

print("running inference...")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=8, do_sample=False)

answer = processor.decode(
    output[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
).strip()

print("answer:", repr(answer))
