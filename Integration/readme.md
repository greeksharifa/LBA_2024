# 실행 방법

환경 및 파일 이름 등은 변경될 수 있음
- 현재 환경은 각각
    - KAIST: `conda activate LBA_2023_KAIST`
    - 경희대: `conda activate LBA_2023`
    - 서울대: `conda activate lavis`
    - 로 가능(LBA ywjang 서버에서)

## run.sh
아래 코드는 예시이며 변경될 수 있음. (현재 작동하는 코드가 아닐 수 있음)
```bash
# git root에서 실행
root_path=$(pwd)
# output_kaist.json 생성
conda activate LBA_2023_KAIST
cd LBA-ARVQA/3rd_year/
python main.py                      # 또는 python LBA_Integration/kaist.py
conda deactivate
cd $root_path

# output_khu.json 생성
conda activate LBA_2023
cd LBA-DramaQG/2024/
python main.py                      # 또는 python LBA_Integration/khu.py
conda deactivate
cd $root_path

# output_snu.json 생성
conda activate lavis
cd LBA-SNU/
python LBA-SNU/eval_dramaqa_sq.py --cfg-path DramaQA_eval.yaml              # 또는 python LBA_Integration/snu.py
```

---

# 입출력 명세(2024년도)

# 1차 개발 - KAIST output (1/22)
LBA_2024/Integration/LBA_2023/LBA-ARVQA/2nd_year/ 폴더 접근
python AnotherMissOh_test.py
실행시 LBA_2024/output_KAIST.json 생성

# KAIST sample
[
    {
        "qid": 3205,
        "question": "How is the relationship between Haeyoung1 and Dokyung when the two hug and kiss each other?",
        "answerability": [
            "unanswerable"
        ],
        "prediction": [
            {
                "['relationship']": false
            }
        ],
        "vid": "AnotherMissOh14_001_0000"
    },
    {
        "qid": 3200,
        "question": "Why did Haeyoung1 lean against the wall when Haeyoung1 was walking in the alley with Dokyung?",
        "answerability": [
            "unanswerable"
        ],
        "prediction": [
            {
                "['wall']": false,
                "['alley']": false
            }
        ],
        "vid": "AnotherMissOh14_001_0000"
    },
    ...
]


# 1차 개발 - KHU output (1/22)
LBA_2024/Integration/LBA_2023/LBA-DramaQG/2023/ 폴더 접근
python run_inference.py --cache_dir="/data2/blip2-flan-t5-xxl" --image_dir="./data/AnotherMissOh/AnotherMissOh_images/"
실행시 LBA_2024/output_KHU.json 생성

# KHU sample
[
    {
        "qid": 3205,
        "vid": "AnotherMissOh14_001_0000",
        "main_question": "How is the relationship between Haeyoung1 and Dokyung when the two hug and kiss each other?",
        "sub_questions": [
            "What is the relationship between Haeyoung1 and Dokyung when the two hug and kiss"
        ]
    },
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

## KAIST 최종 명세

다음 코드를 실행하면 돌아갈 수 있게 코드 업데이트 해주세요. (`main.py` 파일명은 바꾸셔도 됩니다.)

```bash
python LBA-ARVQA/3rd_year/main.py
```


## KHU 최종 명세

다음 코드를 실행하면 돌아갈 수 있게 코드 업데이트 해주세요. (`main.py` 파일명은 바꾸셔도 됩니다.)

```bash
python LBA-DramaQG/2024/main.py
```
