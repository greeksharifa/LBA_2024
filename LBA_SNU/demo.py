import argparse
import os
from glob import glob

import gradio as gr
import numpy as np
import requests
import torch
from gradio_client import Client
from PIL import Image
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

args = None
qg_client = Client("http://127.0.0.1:7866")


def get_info_from_vid(vid):
    episode_id = vid[13:15]
    scene_id = vid[16:19]
    shot_id = vid[20:]

    return episode_id, scene_id, shot_id


def get_image_from_vid(image_dir, vid):
    episode_id, scene_id, shot_id = get_info_from_vid(vid)

    if shot_id == "0000":
        image_path = os.path.join(
            image_dir, f"AnotherMissOh{episode_id}", f"{scene_id}", "**/*.jpg"
        )
        image_list = glob(image_path, recursive=True)

    else:
        image_path = os.path.join(
            image_dir, f"/AnotherMissOh{episode_id}/{scene_id}/{shot_id}/*.jpg"
        )
        image_list = glob(image_path)

    return image_list


class SNUModule:
    def __init__(self, args):
        self.args = args
        if args.cache_dir:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                torch_dtype=torch.float16,
                # load_in_half_prec=True,
                # load_in_4bit=True,
            )
            self.processor = InstructBlipProcessor.from_pretrained(
                args.model_name_or_path, cache_dir=args.cache_dir
            )
        else:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                args.model_name_or_path
            )
            self.processor = InstructBlipProcessor.from_pretrained(
                args.model_name_or_path
            )
        self.device = "cuda"
        self.model.to("cuda")


    def _get_outputs(self, image, text):
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )

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
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].strip()

        return generated_text

    def get_sub_answer(self, image, sub_q):
        print("sub_q:", sub_q)
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
        prompt += "main_answer: "

        prompt = main_question
        for i, (sub_q, sub_a) in enumerate(zip(sub_q_list, sub_a_list)):
            prompt += f"{sub_q} {sub_a}. \n"
        prompt += main_question

        print("prompt:", prompt)

        return self._get_outputs(image, prompt)


def generate_sub_question() -> str:
    sub_question = qg_client.predict("9936297", api_name="/predict")

    return sub_question


def generate_sub_answer(sub_question: str) -> str:
    # TODO: image
    # url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image = Image.open("/data1/AnotherMissOh/images/AnotherMissOh14/036/1254/IMAGE_0000103459.jpg").convert("RGB")

    sub_answer = snu_module.get_sub_answer(image, sub_question)
    return sub_answer


def generate_main_answer(
    main_question: str,
    sub_question: str = "",
    sub_answer: str = "",
):
    # TODO: image
    # url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image = Image.open("/data1/AnotherMissOh/images/AnotherMissOh14/036/1254/IMAGE_0000103459.jpg").convert("RGB")

    main_answer = snu_module.get_main_answer(
        image,
        main_question,
        [sub_question],
        [sub_answer],
    )
    return main_answer


def load_on_gpu(args):
    global snu_module
    snu_module = SNUModule(args)


def add_args(parser):
    parser.add_argument(
        "--model_name_or_path", type=str, default="Salesforce/instructblip-vicuna-13b"
    )  # "Salesforce/blip2-flan-t5-xxl"
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--port", type=int, default=7862)
    # parser.add_argument("--image_dir", type=str, required=True)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    load_on_gpu(args)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Upload Video")
            with gr.Column():
                main_question = gr.Textbox(label="Main Question")
                main_answer_wo_subqa = gr.Textbox(label="Main Answer")
                sub_question = gr.Textbox(
                    label="Sub Question",
                    interactive=False,
                )
                sub_answer = gr.Textbox(
                    label="Sub Answer",
                    interactive=False,
                )
                main_answer = gr.Textbox(
                    label="Main Answer with Sub QA",
                    interactive=False,
                )

                main_answer_wo_subqa.change(
                    fn=generate_sub_question,
                    inputs=[],
                    outputs=[sub_question],
                )
                sub_question.change(
                    fn=generate_sub_answer,
                    inputs=[sub_question],
                    outputs=[sub_answer],
                )
                sub_answer.change(
                    fn=generate_main_answer,
                    inputs=[main_question, sub_question, sub_answer],
                    outputs=[main_answer],
                )
        btn = gr.Button("Answer Me")
        btn.click(
            fn=generate_main_answer,
            inputs=[main_question],
            outputs=[main_answer_wo_subqa],
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=True,
    )

