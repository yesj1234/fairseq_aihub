#! /usr/bin/env bash 

LOCAL_RANK=0,1,2,3 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node 4 \
--use-env run_speech_recognition_ctc.py \
run_speech_recognition_ctc.json \


