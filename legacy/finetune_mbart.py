#mBART large cc25 finetuning 
# required installation 
# !pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# pip install transformers["ja"] numpy pandas sentencepiece fairseq

# Create training data for tokenizer 
res = []
for line in open('split/train', 'r', encoding='utf-8'):
    text = line.split('\t')
    text = [t.rstrip('\n') for t in text]
    res.extend(text)
for line in open('split/dev', 'r', encoding='utf-8'):
    text = line.split('\t')
    text = [t.rstrip('\n') for t in text]
    res.extend(text)
for line in open('split/test', 'r', encoding='utf-8'):
    text = line.split('\t')
    text = [t.rstrip('\n') for t in text]
    res.extend(text)

print(len(res))
with open('tmp.txt', 'w') as f:
    for d in res:
        f.write("%s\n" % d)


import sentencepiece as spm
spm.SentencePieceTrainer.Train("--input=tmp.txt --model_prefix=new_spm_model --vocab_size=64000 --vocabulary_output_piece_score=false --model_type=bpe")

#Formatting vocab 
edited = []
for line in open("new_spm_model.vocab", 'r', encoding='utf-8'):
    if line in ["<unk>\n", "<s>\n", "</s>\n"]:
        continue
    new_line = line.rstrip('\n') + " 1\n"
    edited.append(new_line)

with open('new_dict.txt', 'w') as f:
    for e in edited:
        f.write(e)
        
# download pre trained model
# !wget "https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz"
#Reduce to create a new model 
from fairseq.data import Dictionary
from transformers import (
    MBartForConditionalGeneration, MBartTokenizer, MBartConfig
)
from typing import List
import torch

langs = [
    "ar_AR",
    "cs_CZ",
    "de_DE",
    "en_XX",
    "es_XX",
    "et_EE",
    "fi_FI",
    "fr_XX",
    "gu_IN",
    "hi_IN",
    "it_IT",
    "ja_XX",
    "kk_KZ",
    "ko_KR",
    "lt_LT",
    "lv_LV",
    "my_MM",
    "ne_NP",
    "nl_XX",
    "ro_RO",
    "ru_RU",
    "si_LK",
    "tr_TR",
    "vi_VN",
    "zh_CN"
]

def load_dict(langs: List[str], path: str) -> Dictionary:
    d = Dictionary.load(path)
    for ll in langs:
        d.add_symbol(f"[{ll}]")
    d.add_symbol("<mask>")
    d.add_symbol("<pad>")
    return d



pre_dict = load_dict(langs, "./mbart.cc25.v2/dict.txt")
ft_dict = load_dict(langs, "./new_dict.txt")

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
org_sd = model.state_dict()
resized_sd = model.state_dict()



mapping: List[int] = []
for i in range(len(ft_dict)):
    word = ft_dict[i]
    mapping.append(pre_dict.index(word))

for name in ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "model.shared.weight", "lm_head.weight"]:
    pre_tensor: torch.Tensor = org_sd[name]
    ft_tensor = torch.zeros(
        [len(ft_dict), 1024], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
    )
    for ft_i, pre_i in enumerate(mapping):
        ft_tensor[ft_i] = pre_tensor[pre_i]
    resized_sd[name] = ft_tensor
resized_sd["final_logits_bias"] = resized_sd["final_logits_bias"][:, :len(ft_dict)]

config = MBartConfig.from_pretrained("facebook/mbart-large-cc25")
config.vocab_size = len(ft_dict)
print(config)
new_model = MBartForConditionalGeneration.from_pretrained(None, config=config, state_dict=resized_sd)
new_model.save_pretrained("./reduced_model")

#Preparation of Tokenizer 
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
tokenizer.save_pretrained("./reduced_model")

# Overwrite the model file with the one you created earlier 
# !mv ./new_spm_model.model ./reduced_model/sentencepiece.bpe.model

model = MBartForConditionalGeneration.from_pretrained("./reduced_model")
tokenizer = MBartTokenizer.from_pretrained("./reduced_model")

from transformers import (
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import numpy as np
import re

result_dir = "./output"

def data_collator(features: list):
    x = [f["translation"]["ja"] for f in features]
    y = [f["translation"]["en"] for f in features]
    inputs = tokenizer(x, return_tensors="pt", padding='max_length', truncation=True, max_length=32)
    with tokenizer.as_target_tokenizer():
        inputs['labels'] = tokenizer(y, return_tensors="pt", padding='max_length', truncation=True, max_length=48)['input_ids']
    return inputs

tokenizer = MBartTokenizer.from_pretrained("./reduced_model", src_lang="ja_XX", tgt_lang="en_XX")
tokenizer.save_pretrained(result_dir)

train_data = []
eval_data = []

for line in open("./split/train", "r", encoding='utf-8'):
    text = line.split('\t')
    train_data.append(
        {"translation": {
            "ja": text[1].rstrip('\n'),
            "en": text[0].rstrip('\n')
        }}
    )
print(f"train_data size: {len(train_data)}")

for line in open("./split/dev", "r", encoding='utf-8'):
    text = line.split('\t')
    eval_data.append(
        {"translation": {
            "ja": text[1].rstrip('\n'),
            "en": text[0].rstrip('\n')
        }}
    )
print(f"eval_data size: {len(eval_data)}")

import numpy as np
import evaluate 
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}




batch_size = 1
learning_rate = 3e-5
epochs = 1

model = MBartForConditionalGeneration.from_pretrained("./reduced_model")

args = Seq2SeqTrainingArguments(output_dir=result_dir,
                                do_train=True,
                                do_eval=True,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                learning_rate=learning_rate,
                                num_train_epochs=epochs,
                                evaluation_strategy="epoch",
                                )

trainer = Seq2SeqTrainer(model=model,
                         args=args,
                         data_collator=data_collator,
                         train_dataset=train_data,
                         eval_dataset=eval_data,
                         )

trainer.train()
trainer.save_model(result_dir)


# Inference 
model = MBartForConditionalGeneration.from_pretrained("./output")
tokenizer = MBartTokenizer.from_pretrained("./output")

sentence = "おはよう"
inputs = tokenizer(sentence, return_tensors="pt")
translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"], early_stopping=True, max_length=48)
pred = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print(f"日本語 - {sentence}: English - {pred}")