export CUDA_VISIBLE_DEVICES=0,1,2,3,4
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# vicuna 7b
# TRANSFORMERS_VERBOSITY=info

port_value_file="./port_value_file.txt"

port_value=$(<$port_value_file)
echo "old: $port_value"

port_value=$(($port_value + 1))
echo "new port_value: $port_value"

echo ${port_value} > ${port_value_file}

NCCL_P2P_DISABLED=1 CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.run --nproc_per_node=5 --master_port=$port_value test.py --cfg-path yamls/DramaQA_eval_sq_finetune.yaml
# NCCL_P2P_DISABLED=1 CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.run --nproc_per_node=5 --master_port=50505 test.py --cfg-path yamls/DramaQA_eval_sq_finetune.yaml