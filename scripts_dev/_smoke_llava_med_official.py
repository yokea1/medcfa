import sys
import torch
from PIL import Image

sys.path.insert(0, "/home/217885@student.upm.edu.my/LLaVA-Med")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

MODEL = "/home/217885@student.upm.edu.my/medcfa/cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2"
IMAGE = "/home/217885@student.upm.edu.my/medcfa/data/chexlocalize/raw/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"

print("loading official LLaVA-Med...")
model_name = "llava_mistral"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL,
    model_base=None,
    model_name=model_name,
    device_map="auto",
)

model.eval()

image = Image.open(IMAGE).convert("RGB")
image_tensor = process_images([image], image_processor, model.config)[0]
image_tensor = image_tensor.to(dtype=torch.float16, device=model.device)

question = (
    "You are a radiologist reviewing a chest X-ray. "
    "Answer with exactly one word: Yes or No. "
    "Question: Is there any sign of Cardiomegaly in this image?"
)

prompt = DEFAULT_IMAGE_TOKEN + "\n" + question

conv = conv_templates["plain"].copy()
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    IMAGE_TOKEN_INDEX,
    return_tensors="pt",
).unsqueeze(0).to(model.device)

print("running inference...")
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0),
        do_sample=False,
        max_new_tokens=32,
        use_cache=True,
    )

answer = tokenizer.decode(output_ids[0], skip_special_tokens=False).strip()
print("answer:", repr(answer))
