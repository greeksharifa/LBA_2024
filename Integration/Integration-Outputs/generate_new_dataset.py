import json
from copy import deepcopy

def main():
    sub_answer_path = "/home/ywjang/LBA_2024/Integration/Integration-Outputs/output_sub_answer_240122_182058.json"
    dramaqa_qa_path = "/home/ywjang/LBA_2024/Integration/LBA-ARVQA/2nd_year/dataset/DramaQA/AnotherMissOhQA_val_set.json"
    
    sub_answers = json.load(open(sub_answer_path, 'r'))
    dramaqa_datas = json.load(open(dramaqa_qa_path, 'r'))

    main_qas = {}
    for dramaqa_data in dramaqa_datas:
        tmp_data = deepcopy(dramaqa_data)
        del tmp_data['qid']
        main_qas[dramaqa_data['qid']] = tmp_data

    results = []

    for sub_answer in sub_answers:
        result = deepcopy(sub_answer)
        result.update(main_qas[sub_answer['qid']])
        results.append(result)

    json.dump(results, open('LBA_data.json', 'w'), indent=4)


if __name__ == "__main__":
    main()
