# MT EXAMPLE

## PREPROCESSING 산출물 DATA

0. **_validation_**

```bash
python3 0.json_validator.py --jsons /path/to/the/folder/containing/json/files
e.g.
python3 0.json_validator.py --jsons ./output
```

1. **_prepare_from_json_mt.py_**

```bash
python3 1.prepare_from_json_mt.py --mt_dest_folder /path/to/the/destination/folder --jsons /path/to/the/folder/containing/jsons
e.g.
python3 1.prepare_from_json_mt.py --mt_dest_folder ./mt_split --jsons ./output/한국어(KO)_영어(EN)

```

2. **_tsv_to_json.py_**

```bash
# preparing json file that will be used in run_training_mbart.py.
# source_lang : en ko ja zh
# target_lang : en ko ja zh
python3 2.tsv_to_json.py --split_path /path/to/the/folder/containing/splits.tsv --source_lang source_lang --target_lang target_lang
e.g.
python3 2.tsv_to_json.py --split_path ./mt_split --source_lang ko --target_lang en
```

3. **_preparing the data for model pruning_**. [tackling OOM issue while training with GPU](https://github.com/facebookresearch/fairseq/issues/2120)

   download the base model mbart.cc25.v2. the folder contains model.pt, dict.txt, sentence.bpe.model

   ```
   wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz
   tar -xzvf mbart.CC25.tar.gz
   ```

   **FIRST**, generate corpus data for mbart.  
   FROM train.tsv TO train.ko train.en  
   FROM test.tsv TO test.ko test.en  
   FROM validation.tsv TO validation.ko validation.en

   ```bash
   # the script is in mt_example/utils
   python3 corpus_gen_for_mbart.py --splits /path/to/the/folder/containing/splits --source_lang source_lang --target_lang target_lang
   e.g.
   python3 corpus_gen_for_mbart.py --splits ./mt_split --source_lang ko --target_lang en
   ```

   **SECOND**, encode the generated corpus with sentencepiece

   FROM train.ko train.en TO train.spm.ko train.spm.en  
   FROM test.ko test.en TO test.spm.ko test.spm.en  
   FROM validatioin.ko validatioin.en TO validatioin.spm.ko validatioin.spm.en  
   BEAWARE that the length of inputs and outputs must match.

   ```bash
   # the script is in mt_example/utils as well
   python3 spm_encode.py --model /path/to/the/saved/model --inputs /path/to/train.ko /path/to/train.en --outputs /path/to/train.spm.ko /path/to/train.spm.en --min_length 10 --max_length 512
   e.g.
   export SPLITS_DIR=/home/ubuntu/contents/한국어_영어
   python3 spm_encode.py --model ./sentence.bpe.model --inputs $SPLITS_DIR/train.ko $SPLITS_DIR/train.en $SPLITS_DIR/test.ko $SPLITS_DIR/test.en $SPLITS_DIR/validation.ko $SPLITS_DIR/validation.en --outputs $SPLITS_DIR/train.spm.ko $SPLITS_DIR/train.spm.en $SPLITS_DIR/test.spm.ko $SPLITS_DIR/test.spm.en $SPLITS_DIR/validation.spm.ko $SPLITS_DIR/validation.spm.en
   export SPLITS_DIR=/home/ubuntu/contents/한국어_일본어
   python3 spm_encode.py --model ./sentence.bpe.model --inputs $SPLITS_DIR/train.ko $SPLITS_DIR/train.ja $SPLITS_DIR/test.ko $SPLITS_DIR/test.ja $SPLITS_DIR/validation.ko $SPLITS_DIR/validation.ja --outputs $SPLITS_DIR/train.spm.ko $SPLITS_DIR/train.spm.ja $SPLITS_DIR/test.spm.ko $SPLITS_DIR/test.spm.ja $SPLITS_DIR/validation.spm.ko $SPLITS_DIR/validation.spm.ja
   export SPLITS_DIR=/home/ubuntu/contents/한국어_중국어
   python3 spm_encode.py --model ./sentence.bpe.model --inputs $SPLITS_DIR/train.ko $SPLITS_DIR/train.zh $SPLITS_DIR/test.ko $SPLITS_DIR/test.zh $SPLITS_DIR/validation.ko $SPLITS_DIR/validation.zh --outputs $SPLITS_DIR/train.spm.ko $SPLITS_DIR/train.spm.zh $SPLITS_DIR/test.spm.ko $SPLITS_DIR/test.spm.zh $SPLITS_DIR/validation.spm.ko $SPLITS_DIR/validation.spm.zh

   ```

   **THIRD**, build the vocab.txt from encoded spm splits (e.g. train.spm.ko train.spm.en)  
   --langs argument is fixed argument with "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN"  
   generated file will be saved as one dict.txt file

   ```bash
   python3 build.py --corpus-data "/path/to/spm_splits_with_regex" --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output /path/to/the/folder/dict.txt
   e.g.
   export SPLITS_DIR=/home/ubuntu/path/to/the/splits_dir
   python3 build.py --corpus-data "$SPLITS_DIR/*.spm.*" --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output ./ft/dict.txt
   ```

   **FORTH**, Finally prune the model with generated **_dict.txt_** file

   ```bash
   python3 prune_mbart.py --pre-dict /home/ubuntu/mbart.cc25.v2/dict.txt --ft-dict ../ft/dict.txt --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output ../reduced_model
   ```

   **FIFTH**, Since I'm using transformers library source code [run_translation.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py), needs to load the correct model and configuration and tokenizer.
   the model, config, and tokenizer can be loaded as follows

   ```python
   from transformers import (
       AutoConfig,
       AutoTokenizer,
       AutoModelForSeq2SeqLM
   )
   config = AutoConfig.from_pretrained('facebook/mbart-large-cc25')
   tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-cc25')
   model = AutoModelForSeq2SeqLM.from_pretrained(
       '/path/to/the/saved/pruned_model.pt',
       config=config)
   ```

   **LASTLY**, go for training with pruned model with right settings. settings can be changed in run_training_mbart.json

   ```bash
   # mt_example/run_training_mbart.sh
   bash run_training_mbart.sh
   ```

## INFERENCE

# ASR EXAMPLE

## PREPROCESSING 산출물 DATA

0. **_validation_**

```bash
python3 0.json_validator.py --jsons /path/to/the/folder/containing/json/files
e.g.
python3 0.json_validator.py --jsons ./output
```

1. **_prepare_from_json_asr.py_**

```bash
python3 1.prepare_from_json_asr.py --asr_dest_folder /path/to/the/destination/folder --jsons /path/to/the/folder/containing/jsons
e.g.
python3 1.prepare_from_json_asr.py --asr_dest_folder ./asr_split --jsons $SPLITS_DIR
```

2. **_refine_data.py_**

Wav2Vec2 xls-r model

```bash
python3 refine_data.py --tsv_splits_dir /path/to/the/tsv/splits
e.g.
python3 refine_data.py --tsv_splits_dir ../asr_split
```

3. export the tsv file path and audio folder path for sample_speech.py to correctly load the data from local.

```bash
export DATA_DIR=/path/to/the/refined_splits
export AUDIO_DIR=/path/to/the/audio/folder
e.g.
export DATA_DIR=/home/ubuntu/my_asr/cycle0/asr_example/asr_split
export AUDIO_DIR=/home/ubuntu/output
```

4. set configurations before running the training script. possible arguments can be found in run_speech_recognition_ctc.sh. for example

```json
{
  "model_name_or_path": "facebook/wav2vec2-large-xlsr-53",
  "overwrite_output_dir": true,
  "freeze_feature_encoder": true,
  "attention_dropout": 0.1,
  "hidden_dropout": 0.1,
  "feat_proj_dropout": 0.1,
  "mask_time_prob": 0.3,
  "mask_feature_length": 64,
  "layerdrop": 0.1,
  "ctc_loss_reduction": "mean",
  "dataset_name": "./sample_speech.py",
  "train_split_name": "train",
  "audio_column_name": "audio",
  "text_column_name": "target_text",
  "eval_metrics": ["cer"],
  "unk_token": "[UNK]",
  "pad_token": "[PAD]",
  "word_delimiter_token": "|",
  "output_dir": "./ko-xlsr",
  "do_train": true,
  "do_predict": true,
  "evaluation_strategy": "steps",
  "eval_steps": 1000,
  "per_device_train_batch_size": 2,
  "per_device_eval_batch_size": 2,
  "gradient_accumulation_steps": 2,
  "num_train_epochs": 50,
  "save_strategy": "epoch",
  "logging_strategy": "epoch",
  "learning_rate": 5e-4,
  "warmup_steps": 500,
  "save_total_limit": 1,
  "group_by_length": true,
  "fp16": true,
  "max_duration_in_seconds": 10,
  "chars_to_ignore": [
    ",",
    "?",
    "!",
    "%",
    "'",
    "~",
    ":",
    "/",
    "(",
    ")",
    ".",
    "·",
    "\u001c",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "@"
  ]
}
```

5. run the training shell script

```bash
bash run_speech_recognition_ctc.bash
```
