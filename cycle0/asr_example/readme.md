# Multi gpu 환경에서 trainer api 사용하기.
1. run_training_gpu_asr.json 
model_args, training_args, data_args 관련된 세팅 하나의 json 파일로 묶어서 관리.
```json
{
    "model_name_or_path": "facebook/wav2vec2-large-xlsr-53" ,
    "overwrite_output_dir": true,
    "freeze_feature_encoder": true,
    "attention_dropout": 0.1 ,
    "hidden_dropout": 0.1 ,
    "feat_proj_dropout": 0.1 ,
    "mask_time_prob" :0.1 ,
    "layerdrop" :0.1 ,
    "ctc_loss_reduction": "mean" ,
    "dataset_name": "mozilla-foundation/common_voice_11_0" ,
    "dataset_config_name":  "ja" ,
    "train_split_name": "train" ,
    "eval_split_name": "validation" ,
    "audio_column_name": "audio" ,
    "text_column_name": "sentence" ,
    "eval_metrics": "cer" ,
    "unk_token": "[UNK]",
    "pad_token": "[PAD]" ,
    "word_delimiter_token": "|" ,
    "output_dir": "./wav2vec2-large-xlsr-jp-test0818_hiragana" ,
    "do_train": true,
    "do_eval": true,
    "do_predict": true,
    "evaluation_strategy":"steps",
    "per_device_train_batch_size": 16 ,
    "per_device_eval_batch_size": 8 ,
    "gradient_accumulation_steps" :2 ,
    "num_train_epochs" :50 ,
    "save_strategy" :"epoch" ,
    "logging_strategy": "epoch" ,
    "learning_rate" :1e-4 ,
    "warmup_steps" :1500 ,
    "save_total_limit" :2 ,
    "group_by_length" :true
}
```

2. bash로 python script 실행 하기 혹은 그냥 python 명령어 실행.
```bash
LOCAL_RANK=0,1,2,3 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node 4 \
--use-env run_training.py \
run_training_gpu_asr.json \
--chars_to_ignore [\,\?\.\!\-\;\:\"\“\‘\”\ ‘、。．！，・―─~｢｣『』〆｡\\\\※\[\]\{\}「」〇？…]
```