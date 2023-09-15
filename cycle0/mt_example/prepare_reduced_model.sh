#! /usr/bin/env bash 

# prepare reduce model for mbart
cd ~
export PWD=$(pwd)
export PREPARE_FROM_JSON_MT=$PWD/my_asr/cycle0/mt_example/1.prepare_from_json_mt.py
export TSV_TO_JSON=$PWD/my_asr/cycle0/mt_example/utils/2.tsv_to_json.py
export CORPUS_GEN_FOR_MBART=$PWD/my_asr/cycle0/mt_example/utils/corpus_gen_for_mbart.py
export SPM_ENCODE=$PWD/my_asr/cycle0/mt_example/utils/spm_encode.py
export BUILD_VOCAB=$PWD/my_asr/cycle0/mt_example/utils/build_vocab.py
export PRUNE_MODEL=$PWD/my_asr/cycle0/mt_example/utils/prune_model.py

export CURRENT_DIR=$PWD/my_asr/cycle0/mt_example
# export SPLITS_DIR=$PWD/path/to/source-target/jsons

# python3 PREPARE_FROM_JSON_MT --mt_dest_file $PWD/path/to/soruce-target/jsons/ \
# --jsons $PWD/path/to/source-target/jsons

# python3 CORPUS_GEN_FOR_MBART --splits $SPLITS_DIR --source_lang ko --target_lang en

# python3 SPM_ENCODE --model $PWD/mbart.cc25.v2/sentence.bpe.model \
# --inputs $SPLITS_DIR/train.ko $SPLITS_DIR/train.en $SPLITS_DIR/test.ko \
# $SPLITS_DIR/test.en $SPLITS_DIR/validation.ko $SPLITS_DIR/validation.en \
# --outputs $SPLITS_DIR/train.spm.ko $SPLITS_DIR/train.spm.en \
# $SPLITS_DIR/test.spm.ko $SPLITS_DIR/test.spm.en $SPLITS_DIR/validation.spm.ko \
# $SPLITS_DIR/validation.spm.en

# python3 BUILD_VOCAB --corpus-data "$SPLITS_DIR/*.spm.*" \
# --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN \
# --output $PWD/my_asr/cycle0/mt_example/ft/dict.txt

# python3 PRUNE_MODEL --pre-dict $PWD/mbart.cc25.v2/dict.txt \
# --ft-dict $PWD/my_asr/ft/dict.txt \
# --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN \
# --output $PWD/my_asr/cycle0/mt_example/reduced_model
