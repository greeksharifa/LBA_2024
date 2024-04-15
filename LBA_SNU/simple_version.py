import os
import argparse

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from glob import glob


def get_info_from_vid(vid):
    episode_id = vid[13:15]
    scene_id = vid[16:19]
    shot_id = vid[20:]

    return episode_id, scene_id, shot_id


def get_image_from_vid(image_dir, vid):
    episode_id, scene_id, shot_id = get_info_from_vid(vid)
    
    if shot_id == "0000":
        image_path = os.path.join(image_dir, f"AnotherMissOh{episode_id}", f"{scene_id}", "**/*.jpg")
        image_list = glob(image_path, recursive=True)
    
    else:
        image_path = os.path.join(image_dir, f"/AnotherMissOh{episode_id}/{scene_id}/{shot_id}/*.jpg")
        image_list = glob(image_path)
    
    return image_list


class SNUModule:
    def __init__(self, args):
        self.args = args
        if args.cache_dir:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir,
                                                                              # load_in_half_prec=True,
                                                                              load_in_4bit=True)
            self.processor = InstructBlipProcessor.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(args.model_name_or_path)
            self.processor = InstructBlipProcessor.from_pretrained(args.model_name_or_path)

        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        
    def _get_outputs(self, image, text):
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        for k, v in inputs.items():
            print(f"{k}: {v.shape}")
        
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        return generated_text
        
        
    def get_sub_answer(self, image, sub_q):
        print('sub_q:', sub_q)
        return self._get_outputs(image, sub_q)
    
    
    def get_main_answer(self, image, main_question, sub_q_list, sub_a_list):
        prompt = f"Instructions: Given a picture, main_question and Q&A results that will help answer the main question. "
        prompt += "Our goal is to answer the main_question. "
        prompt += f"main_question: {main_question} "
        for i, (sub_q, sub_a) in enumerate(zip(sub_q_list, sub_a_list)):
            prompt += f"{sub_q} {sub_a}. "
            # prompt += f"sub_q_{i+1}: {sub_q} "
            # prompt += f"sub_a_{i+1}: {sub_a}. "
        prompt += f"main_question: {main_question} "
        # prompt += "main_answer: "
        
        print('prompt:', prompt)
        
        return self._get_outputs(image, prompt)


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    
    snu_module = SNUModule(args)
    
    # TODO: gradio
    
    
    # 아래는 데모용 코드
    url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # sub_q = "What is unusual about this image?"
    sub_q = "Write a detailed caption for this image."
    print('get_sub_answer:', snu_module.get_sub_answer(image, sub_q))
    
    print('get_main_answer:', snu_module.get_main_answer(image, "What is unusual about this image?",
                                                         ['what is man doing?'],
                                                         ['ironing']))
    


def add_args(parser):
    parser.add_argument("--model_name_or_path", type=str, default="Salesforce/instructblip-vicuna-7b")# "Salesforce/blip2-flan-t5-xxl"
    parser.add_argument("--cache_dir", type=str)
    # parser.add_argument("--image_dir", type=str, required=True)
    
    return parser


if __name__ == "__main__":
    main()
    
