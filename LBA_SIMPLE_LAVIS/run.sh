# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,3,4,5 CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.run --nproc_per_node=4 --master_port=50505 test.py --cfg-path yamls/DramaQA_eval_sq_finetune.yaml
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.run --nproc_per_node=4 --master_port=50505 test.py --cfg-path projects/LBA/eval/DramaQA_eval_sq_finetune.yaml

