from datasets import load_dataset, Audio, concatenate_datasets, load_metric
import pykakasi
import re 
import MeCab
import json
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import torch 
import numpy as np
import torchaudio
from datetime import datetime 

#1.Prepare Dataset.
covost_jp_train = load_dataset("mozilla-foundation/common_voice_11_0", "ja", split="train")
covost_jp_test = load_dataset("mozilla-foundation/common_voice_11_0", "ja", split="test")
covost_jp_val = load_dataset("mozilla-foundation/common_voice_11_0", "ja", split="validation")

covost_jp_train = covost_jp_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
covost_jp_test = covost_jp_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
covost_jp_val = covost_jp_val.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

# covost_jp_train = covost_jp_train.cast_column("audio", Audio("sampling_rate", 16000))
# covost_jp_test = covost_jp_test.cast_column("audio", Audio("sampling_rate", 16000))
# covost_jp_val = covost_jp_val.cast_column("audio", Audio("sampling_rate", 16000))

#2. Preprocess datasets.
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\‘\”\�‘、。．！，・―─~｢｣『』〆｡\\\\※\[\]\{\}「」〇？…]'
wakati = MeCab.Tagger("-Owakati")
kakasi = pykakasi.kakasi()
kakasi.setMode("J","H")      # kanji to hiragana
kakasi.setMode("K","H")      # katakana to hiragana
converter = kakasi.getConverter()

month_and_date = f"{datetime.now().month}{datetime.now().day}"
now = datetime.now()
vocabPath = f"./vocab_jp_hiragana_{now.strftime('%m%d')}.json"

FULLWIDTH_TO_HALFWIDTH = str.maketrans(
    '　０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ！゛＃＄％＆（）＊＋、ー。／：；〈＝〉？＠［］＾＿‘｛｜｝～',
    ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&()*+,-./:;<=>?@[]^_`{|}~',
)
def fullwidth_to_halfwidth(s):
    return s.translate(FULLWIDTH_TO_HALFWIDTH)

def preprocess_text(batch):
    batch["sentence"] = fullwidth_to_halfwidth(batch["sentence"])
    batch["sentence"] = re.sub(chars_to_ignore_regex,' ', batch["sentence"]).lower()  #remove special char
    batch["sentence"] = wakati.parse(batch["sentence"])                              #add space
    batch["sentence"] = converter.do(batch["sentence"])                                   #covert to hiragana
    batch["sentence"] = " ".join(batch["sentence"].split())+" "                    #remove multiple space    
    return batch

def createVocabList(dataset):
    def extract_all_chars(batch):
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    
    dataset=dataset.map(preprocess_text)
    vocab_train = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names) #concat all text
    vocab_list = list(set(vocab_train["vocab"][0]) )      #convert to set
    vocab_list = sorted(vocab_list)
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}   # convert to dict
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocabPath, 'w')as vocab_file:
        json.dump(vocab_dict, vocab_file)

    print(vocab_dict)
    
    
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = torchaudio.functional.resample(speech_array, sampling_rate, 16000)[0].numpy()
    batch["target_text"] = batch["sentence"]
    return batch

createVocabList(concatenate_datasets([covost_jp_train, covost_jp_val]))

datasets_train = concatenate_datasets([covost_jp_train, covost_jp_val])
datasets_train = datasets_train.map(preprocess_text)
datasets_train = datasets_train.map(speech_file_to_array_fn, remove_columns=datasets_train.column_names)

datasets_test = concatenate_datasets([covost_jp_test])
datasets_test = datasets_test.map(preprocess_text)
datasets_test = datasets_test.map(speech_file_to_array_fn, remove_columns=datasets_test.column_names)

#3. initialize required instances to train wav2vec2 model. Tokenizer, feature_extractor, processor, trianer, training arguments, and custom data collator class.
tokenizer = Wav2Vec2CTCTokenizer(vocabPath, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def prepare_dataset(batch):    
    batch["input_values"] = processor(batch["speech"], sampling_rate=16000, padding=True).input_values
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

datasets_train = datasets_train.map(prepare_dataset, batch_size=4, batched=True)
datasets_test = datasets_test.map(prepare_dataset, batch_size=4, batched=True)


wer_metric = load_metric("wer")


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

#4. put them all together. 
data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True)

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
model.freeze_feature_extractor()

model_temp_output_dir = f"./wav2vec2-large-xlsr-jp-test{now.strftime('%m%d')}_hiragana"
training_args = TrainingArguments(
  output_dir=model_temp_output_dir,  
  group_by_length=True,
  per_device_train_batch_size=3,
  gradient_accumulation_steps=2,
  per_device_eval_batch_size=2,
  num_train_epochs=50,
  fp16=True,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  logging_strategy="epoch",
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=1,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=datasets_train,
    eval_dataset=datasets_test,
    tokenizer=processor.feature_extractor,
)

if __name__ == "__main__":
    import time
    import torch
    
    start = time.time()
    trainer.train()
    end = time.time() 
    print("-"*30 + f"{end-start} Seconds" + "-"*30)
    
    