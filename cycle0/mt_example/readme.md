# Multi gpu 환경에서 trainer api 사용하기.
1. run_training_gpu_mt.json 
model_args, training_args, data_args 관련된 세팅 하나의 json 파일로 묶어서 관리.
```json
{
    "model_name_or_path": "facebook/mbart-large-50-many-to-many-mmt",
    "source_lang": "en_XX",
    "target_lang": "ko_KR",
    "dataset_name": "opus100",
    "dataset_config_name": "en-ko",
    "pad_to_max_length": true, 
    "output_dir": "./tst-translation-output",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 16,
    "overwrite_output_dir": true,
    "do_train": true, 
    "do_eval": true,
    "predict_with_generate": true
}
```

2. bash로 python script 실행 하기 혹은 그냥 python 명령어 실행.
```bash
LOCAL_RANK=0,1,2,3 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node 4 \
--use-env run_translation.py \
run_translation_gpu_mt.json
```