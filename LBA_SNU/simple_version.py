import os
import json
import argparse

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from glob import glob

from utils import *


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
        
        
    def _get_outputs(self, images, text):
        inputs = self.processor(images=images, text=text, return_tensors="pt").to(self.device)
        # for k, v in inputs.items():
            # print(f"{k}: {v.shape}")
        
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
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
        
        
    def get_sub_answer(self, images, sub_q):
        # print('sub_q:', sub_q)
        return self._get_outputs(images, sub_q)
    
    
    def get_main_answer(self, images, main_question, sub_q_list, sub_a_list):
        prompt = f"Instructions: Given a picture, main_question and Q&A results that will help answer the main question. "
        prompt += "Our goal is to answer the main_question. "
        # prompt += f"\nmain_question: {main_question} "
        prompt += "\nsub Q&A results:"
        for i, (sub_q, sub_a) in enumerate(zip(sub_q_list, sub_a_list)):
            prompt += f"\n{sub_q} {sub_a}"
            # prompt += f"sub_q_{i+1}: {sub_q} "
            # prompt += f"sub_a_{i+1}: {sub_a}. "
        prompt += f"\nNow answer the main_question: {main_question} "
        # prompt += "main_answer: "
        
        print('prompt:', prompt)
        
        return self._get_outputs(images, prompt)


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    
    snu_module = SNUModule(args)
    
    # TODO: gradio
    
    
    # sample sub q generate    
    image = Image.open("demo/IMAGE_0000004657.jpg").convert("RGB")
    sub_q = "What are Jeongsuk and Deogi wearing?"
    print('sample sub_q:', sub_q)
    print('sample sub_a:', snu_module.get_sub_answer(image, sub_q))
    
    
    # answer the main_question
    samples = json.load(open(args.output_KHU_path, 'r'))
    
    for i, sample in enumerate(samples):
        main_q = sample['main_question'] # ex. "Why did Sungjin tell Haeyoung1 not to talk with flowers ?"
        sub_qs = sample['sub_questions'] # ex, ["What kind of flowers did Sungjin tell Haeyoung1 not to talk with?"]
        vid = sample['vid']              # ex. "AnotherMissOh17_014_0000"
        
        # image_paths = get_image_path(args, sample)
        image_path = get_image_from_vid(args.root_dir, vid)[0]
        print(f'{i:4d}th | image_path: {image_path}')
        
        # images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        images = Image.open(image_path).convert("RGB")
        
        sub_as = [snu_module.get_sub_answer(images, sub_q) for sub_q in sub_qs]
        main_a = snu_module.get_main_answer(images, main_q, sub_qs, sub_as)
        print('main_a:', main_a, '\n')
        
    
    return
    
    
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
    
    parser.add_argument('--root_dir', type=str, default="/data1/AnotherMissOh/AnotherMissOh_images/")
    parser.add_argument("--output_KHU_path", type=str, default="../output_sample/output_KHU.json")
    # parser.add_argument('--max_vision_num', type=int, default=1)
    
    return parser


if __name__ == "__main__":
    main()
    
