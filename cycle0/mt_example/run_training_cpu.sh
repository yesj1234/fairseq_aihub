python3 -m torch.distributed.launch --nproc_per_node 4 \
--use-env run_training_mbart.py \
run_training_cpu.json