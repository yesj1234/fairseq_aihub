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
python3 1.prepare_from_json_mt.py --mt_dest_file /path/to/the/destination/folder --jsons /path/to/the/folder/containing/jsons
e.g.
python3 1.prepare_from_json_mt.py --mt_dest_file ./mt_split --jsons ./output/한국어(KO)_영어(EN)
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
   [DOWNLOAD the spm model](https://huggingface.co/facebook/mbart-large-cc25/tree/main) file maybe name sentence.bpe.model
   BEAWARE that the length of inputs and outputs must match.

   ```bash
   # the script is in mt_example/utils as well
   python3 spm_encode.py --model /path/to/the/saved/model --inputs /path/to/train.ko /path/to/train.en --outputs /path/to/train.spm.ko /path/to/train.spm.en --min_length 10 --max_length 512
   e.g.
   export SPLITS_DIR=/home/ubuntu/contents/한국어_영어
   python3 spm_encode.py --model ./sentence.bpe.model --inputs $SPLITS_DIR/train.ko $SPLITS_DIR/train.en $SPLITS_DIR/test.ko $SPLITS_DIR/test.en $SPLITS_DIR/validation.ko $SPLITS_DIR/validation.en --outputs $SPLITS_DIR/train.spm.ko $SPLITS_DIR/train.spm.en $SPLITS_DIR/test.spm.ko $SPLITS_DIR/test.spm.en $SPLITS_DIR/validation.spm.ko $SPLITS_DIR/validation.spm.en
   ```

   **THIRD**, build the vocab.txt from encoded spm splits (e.g. train.spm.ko train.spm.en)
   --langs argument is fixed argument with ar*AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
   generated file will be saved as one \*\*\_dict.txt*\*\* file

   ```bash
   python3 build.py --corpus_data "/path/to/spm_splits_with_regex" --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output /path/to/the/folder/dict.txt
   e.g.
   python3 build.py --corpus_data "./ft/*.spm.*" --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output ./ft/dict.txt
   ```

   **FORTH**, Finally prune the model with generated **_dict.txt_** file
   [DOWNlOAD the mbart-large-cc25 base model](https://huggingface.co/facebook/mbart-large-cc25/tree/main) file. Maybe named pytorch_model.bin

   ```bash
   python trim_mbart.py --pre-train-dir /path/to/the/folder/containing/model_to_be_pruned/ --ft-dict /path/to/the/folder/containing/dict.txt --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output /path/to/the/folder/pruned_model
   e.g.
   python trim_mbart.py --pre-train-dir ./mbart.cc25 --ft-dict ./ft/dict.xt --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output ./ft/model.pt
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

# ASR EXAMPLE

## PREPROCESSING 산출물 DATA

0. **_validation_**

```bash
python3 0.json_validator.py --jsons /path/to/the/folder/containing/json/files
e.g.
python3 0.json_validator.py --jsons ./output
```

1. **_prepare_from_json_mt.py_**

```bash
python3 1.prepare_from_json_mt.py --mt_dest_file /path/to/the/destination/folder --jsons /path/to/the/folder/containing/jsons
e.g.
python3 1.prepare_from_json_mt.py --mt_dest_file ./mt_split --jsons ./output/한국어(KO)_영어(EN)
```

2. **_refine_data.py_**

Wav2Vec2 xls-r model

```bash

```
