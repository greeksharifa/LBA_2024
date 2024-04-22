# DramaQA Sub-A 생성

```bash
CUDA_VISIBLE_DEVICES=1 python eval_vqa.py
```

# DramaQA Main-A 생성

- baseline
```bash
python evaluate.py --cfg-path yamls/DramaQA_eval.yaml
```

- LBA-SNU method
```bash
python evaluate.py --cfg-path yamls/DramaQA_eval_sq.yaml
```
