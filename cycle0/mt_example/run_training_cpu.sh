python ./run_translation.py \
    --model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
    --do_train \
    --do_eval \
    --dataset_name opus100 \
    --dataset_config_name en-ko \
    --source_lang en_XX \
    --target_lang ko_KR \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate