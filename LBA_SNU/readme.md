# LBA_SNU

## Installation
```bash
# conda
conda create -n LBA_SNU python=3.8
conda activate LBA_SNU

# lavis
# 프로젝트 바깥 or 원하는 위치에 설치 가능
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .

# resolve version conflict
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install chardet
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 
pip install opencv-python==4.7.0.72
pip install Triton==2.1.0
```

## Run

- args 경로 설정 후 아래 코드 실행

```bash
python simple_version.py
```

참고: args
```
parser.add_argument("--model_name_or_path", type=str, default="Salesforce/instructblip-vicuna-7b")# "Salesforce/blip2-flan-t5-xxl"
parser.add_argument("--cache_dir", type=str)

parser.add_argument('--root_dir', type=str, default="/data1/AnotherMissOh/AnotherMissOh_images/")
parser.add_argument("--output_KHU_path", type=str, default="../output_sample/output_KHU.json")
```

## Data sample

- output_KHU.json

```json
[
    {
        "qid": 13335,
        "vid": "AnotherMissOh17_014_0000",
        "main_question": "Why did Sungjin tell Haeyoung1 not to talk with flowers ?",
        "sub_questions": [
            "What kind of flowers did Sungjin tell Haeyoung1 not to talk with?"
        ]
    },
    {
        "qid": 13648,
        "vid": "AnotherMissOh17_032_0000",
        "main_question": "Who wears striped shirts when Haeyoung2 is talking ?",
        "sub_questions": [
            "What color are the shirts that Haeyoung2 is wearing?"
        ]
    },
    ...
]
```

- AnotherMissOh

```json
[
    {
        "videoType": "scene",
        "answers": [
            "Because Deogi had to call Dokyung.",
            "Because Deogi wanted to go outside.",
            "Because Deogi had to cook food.",
            "Because Deogi wanted to play with Dokyung.",
            "Because Deogi was sleepy after meeting Dokyung."
        ],
        "q_level_logic": 3,
        "qid": 8288,
        "shot_contained": [
            78,
            159
        ],
        "q_level_mem": 3,
        "que": "Why was Deogi in the kitchen?",
        "vid": "AnotherMissOh01_001_0000",
        "correct_idx": 2
    },
    {
        "videoType": "shot",
        "answers": [
            "Jeongsuk and Deogi are wearing a swimsuit.",
            "Jeongsuk and Deogi are wearing a sleeveless shirt.",
            "Jeongsuk and Deogi are wearing a dress.",
            "Jeongsuk and Deogi are wearing an apron.",
            "Jeongsuk and Deogi are wearing a pajama."
        ],
        "q_level_logic": 1,
        "qid": 995219,
        "shot_contained": [
            84
        ],
        "q_level_mem": 2,
        "que": "What are Jeongsuk and Deogi wearing?",
        "vid": "AnotherMissOh01_001_0084",
        "correct_idx": 3
    },
    ...
]
```