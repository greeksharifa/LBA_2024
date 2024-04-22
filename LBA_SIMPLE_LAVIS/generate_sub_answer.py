import os
import json
import argparse
import datetime

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
        image_path = os.path.join(image_dir, f"AnotherMissOh{episode_id}/{scene_id}/{shot_id}/*.jpg")
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
        
        
    def get_sub_answer(self, image, sub_q):
        # print('sub_q:', sub_q)
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

    output_path = os.path.join(args.output_dir, f"output_sub_answer_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.json")
    print('output_path:', output_path)
    
    snu_module = SNUModule(args)
    
    # TODO: gradio
    print("args.demo:", args.demo)
    
    if args.demo:
        # 아래는 데모용 코드
        url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        # sub_q = "What is unusual about this image?"
        sub_q = "Write a detailed caption for this image."
        print('get_sub_answer:', snu_module.get_sub_answer(image, sub_q))
        
        print('get_main_answer:', snu_module.get_main_answer(image, "What is unusual about this image?",
                                                            ['what is man doing?'],
                                                            ['ironing']))
    else:
        outputs = []
        samples = json.load(open(args.question_path, 'r'))
        """        
        # output_KHU.json = generated sub Qs
        [
            {
                "qid": 3200,
                "vid": "AnotherMissOh14_001_0000",
                "main_question": "Why did Haeyoung1 lean against the wall when Haeyoung1 was walking in the alley with Dokyung?",
                "sub_questions": [
                    "What kind of wall is Haeyoung1 leaning against?",
                    "What kind of alley is this?"
                ]
            },
            ...
        ]
        """
        for i, sample in enumerate(samples):
            # if i >= 10:
            #     break
            qid = sample["qid"]
            sub_questions = sample["sub_questions"]
            print("\nqid:", qid)
            print("sub_questions:", sub_questions)

            vid = sample["vid"]
            print("vid:", vid)
            image_list = get_image_from_vid(args.image_path, vid)
            print("image_list:", len(image_list), image_list[:1])
            image = Image.open(image_list[0]).convert("RGB")

            sub_answers = []
            for sub_question in sub_questions:
                sub_answer = snu_module.get_sub_answer(image, sub_question)
                print("sub_answer:", sub_answer)
                sub_answers.append(sub_answer)

            output = sample
            output["sub_answers"] = sub_answers

            outputs.append(output)

        json.dump(outputs, open(output_path, "w"), indent=4)


def add_args(parser):
    parser.add_argument("--model_name_or_path", type=str, default="Salesforce/instructblip-vicuna-7b")# "Salesforce/blip2-flan-t5-xxl"
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--demo", default=False, action="store_true")
    parser.add_argument("--image_path", type=str, default="/data1/AnotherMissOh/AnotherMissOh_images/")
    parser.add_argument("--question_path", type=str, default="../Integration/output_sample/output_KHU.json")
    parser.add_argument("--output_dir", type=str, default="../Integration/Integration-Outputs/")
    # parser.add_argument("--image_dir", type=str, required=True)
    
    return parser


if __name__ == "__main__":
    main()
    


"""
        for i, sample in enumerate(samples):
            # if i >= 3:
                # break
            qid = sample["qid"]
            sub_question = sample["question"]
            print("\nqid:", qid)
            print("sub_question:", sub_question)

            vid = sample["vid"]
            print("vid:", vid)
            image_list = get_image_from_vid(args.image_path, vid)
            print("image_list:", len(image_list), image_list[:1])
            image = Image.open(image_list[0]).convert("RGB")

            sub_answer = snu_module.get_sub_answer(image, sub_question)
            print("sub_answer:", sub_answer)

            outputs.append({
                "qid": qid,
                "vid": vid,
                "sub_question": sub_question,
                "sub_answer": sub_answer,
            })
"""
