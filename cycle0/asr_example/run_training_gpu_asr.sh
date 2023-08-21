#! /usr/bin/env bash 

LOCAL_RANK=0,1,2,3 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node 4 \
--use-env run_training.py \
run_training_gpu_asr.json \
--chars_to_ignore [\,\?\.\!\-\;\:\"\“\‘\”\ ‘、。．！，・―─~｢｣『』〆｡\\\\※\[\]\{\}「」〇？…]

