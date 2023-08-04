from datasets import load_dataset, load_metric, ClassLabel
from transformers import Wav2Vec2CTCTokenizer, Wav2VecFeatureExtractor, Wav2VecProcessor
import random
import re
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import (Wav2VecForCTC, Trainer, TrainingArguments)


kspon = load_dataset("./ksponspeech.py", data_dir="./ksponspeech")

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).lower()
    return batch

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocabs = kspon.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True)

vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

vocab_dict = {value: index for index, value in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
