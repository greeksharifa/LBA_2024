import torch
from PIL import Image
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
raw_image = Image.open("Confusing-Pictures.jpg").convert("RGB")
# display(raw_image.resize((596, 437)))

from lavis.models import load_model_and_preprocess
# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

while True:
    prompt = input()
    model.generate({"image": image, "prompt": prompt}) # "What is unusual about this image?"