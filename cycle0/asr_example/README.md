# Multi gpu 환경에서 trainer api 사용하기.

1. bash로 python script 실행 하기 혹은 그냥 python 명령어 실행.

```bash
LOCAL_RANK=0,1,2,3 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node 4 \
--use-env run_training.py \
run_training_gpu_asr.json \
--chars_to_ignore [\,\?\.\!\-\;\:\"\“\‘\”\ ‘、。．！，・―─~｢｣『』〆｡\\\\※\[\]\{\}「」〇？…]
```
