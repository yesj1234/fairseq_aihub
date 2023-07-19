# This is
A repository for reproducing the result from the [paper](https://arxiv.org/abs/2107.02875).
3 models are suggested in the paper. 
first, base transformer model
second, asr pretrained encoder and transformer decoder trained on top of it.
last, warm up + fine tuning version. Data used for warming up the model is the same data used for asr pretrained version but with pseudo-gold translation scripts included(translated by PAPAGO API).
This repository is for reproducing the second version of the paper.

## prerequisite.
follow the steps to download the kosp2e data and specific fairseq version [here](https://github.com/warnikchow/kosp2e)
download ksponspeech data from [AIHUB](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123)
### Environment setting
1. python >= 3.10 

## How to 
### train asr encoder with kspon data.
1. cd into the folder where kspon data is and run preprocessing_data.py
```
python preprocessing_data.py --data_root $ASR_ROOT
```

make sure the $ASR_ROOT is correct.
```
echo $ASR_ROOT
```

2. run prep_data_kos.py where the dataset folder is located.
```
python prep_data_kos.py --data-root dataset/ --task asr --vocab-type unigram --vocab-size 8000
```

3. train asr with following fairseq cli 
if you have gpu installed.
```
fairseq-train $ASR_ROOT/kr-en \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir $ASR_SAVE_DIR --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```

if you have no gpu
```
fairseq-train $ASR_ROOT/en-de \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir $ASR_SAVE_DIR --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --cpu
```

again make sure you have set the right path to the variable ASR_ROOT and ASR_SAVE_DIR
```
echo $ASR_ROOT
echo $ASR_SAVE_DIR #set any directory you want. 
```


 