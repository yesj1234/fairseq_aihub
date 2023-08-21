# links
## MT
1. ["mBART model checkpoint."](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) 
2. [example scripts from fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart)
3. [Translation task guide huggingface](https://huggingface.co/docs/transformers/tasks/translation)
## ASR 
1. [xlsr english checkpoint](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english)
2. [xlsr japanese checkpoint](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese)
3. [xlsr chinese(zh-cn) checkpoint](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn)
4. [xlsr korean checkpoint](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean)

## References
### huggingface apis
1. [pipeline huggingface](https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/pipelines#transformers.pipeline)
2. [Training args](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)
3. []

## Distributed Training and mixed precision(Multi GPU Settings)
1. [trasformers/huggingface/pytorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch#distributed-training-and-mixed-precision)
1-1. [related guide about distributed training from pytorch](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel)
2. 
# Setting OCI BM GPU instance.
[source from here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions)
Environment:
    Ubuntu20.04LTS
    BM.GPU.A10.4
0. ssh into OCI instance
```bash
ssh -i <private key 파일 위치> opc@<public IP> #for Oracle Linux instance
ssh -i <private key 파일 위치> ubuntu@<public IP> #for Ubuntu instance
```
1. Verify the system has a CUDA-Capable GPU
```bash
lspci | grep -i nvidia
```
2. Verify You Have a Supported Version of Linux
```bash
uname -m && cat /etc/*release
```
3. Verify the system has the gcc installed and cmake installed
```bash
gcc --version
cmake --version
```
4. Verify the System has the correct kernel headers and development packages installed.
```bash
uname -r #check the version of the kernel your system is running
```
5. choose an installation method. (distribution specific packages or distribution independent packages). It is recommended to use the distribution specific packages, where possible.
6. download the NVIDIA CUDA toolkit from [here](https://developer.nvidia.com/cuda-downloads)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```
## setting model training settings and pip libraries
1. install pip
```bash
sudo apt install python3-pip
```
2. covost_ja.py 실행을 위한 필요 라이브러리 설치
```bash
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar xvfz mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
make check
make install
sudo ldconfig
mecab --version
pip install mecab-python3 
pip install unidic-lite # dependecy for mecab tagging
```
```bash
pip install pykakasi # for japanese tokenizing
pip install mecab # for tagging japanese
pip install transformers datasets 
pip install torch torchaudio torchvision
pip install numpy 
pip install jiwer #for loading the WER metric from datasets
```
RuntimeError: failed to load mp3 from ... 에러
```bash 
# pip install soundfile  
pip install librosa
sudo apt-get install sox 
pip install sox 
sudo apt update
sudo apt install ffmpeg
```
install soundfile manually. 
```bash
git clone https://github.com/bastibe/python-soundfile.git
cd python-soundfile/
python build_wheels.py
```

ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.20.1`: Please run `pip install transformers[torch]` or `pip install accelerate -U`
```bash
pip install accelerate -U
```

Multi GPU:
```bash
LOCAL_RANK=0,1,2,3 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node 4 \
--use-env run_speech_recognition_ctc.py \
--model_name_or_path facebook/wav2vec2-large-xlsr-53 \
--overwrite_output_dir \
--freeze_feature_encoder True \
--attention_dropout 0.1 \
--hidden_dropout 0.1 \
--feat_proj_dropout 0.1 \
--mask_time_prob 0.1 \
--layerdrop 0.1 \
--ctc_loss_reduction mean \
--dataset_name mozilla-foundation/common_voice_11_0 \
--dataset_config_name ja \
--train_split_name train \
--eval_split_name validation \
--audio_column_name audio \
--text_column_name sentence \
--eval_metrics cer \
--chars_to_ignore [\,\?\.\!\-\;\:\"\“\‘\”\ ‘、。．！，・―─~｢｣『』〆｡\\\\※\[\]\{\}「」〇？…] \
--unk_token [UNK] \
--pad_token [PAD] \
--word_delimiter_token '|' \
--output_dir ./wav2vec2-large-xlsr-jp-test0818_hiragana \
--do_train --do_eval --do_predict \
--evaluation_strategy steps \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 2 \
--num_train_epochs 50 \
--save_strategy epoch \
--logging_strategy epoch \
--learning_rate 1e-4 \
--warmup_steps 1500 \
--save_total_limit 2 \
--group_by_length True
```
## Monitoring 
1. GPU usage monitoring
```bash
watch -d -n 1 nvidia-smi
```
2. CPU usage monitoring
```bash
top
```
