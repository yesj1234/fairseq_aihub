# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Korean Spontaneous Speech Corpus for Automatic Speech Recognition (KsponSpeech)"""

from os.path import *

import datasets
import re

_DESCRIPTION = "sample data for Cycle 1(6%)"

class KsponSpeech(datasets.GeneratorBasedBuilder):
    """The Korean Spontaneous Speech Corpus for Automatic Speech Recognition (KsponSpeech)"""

    VERSION = datasets.Version("0.1.0")

    @property
    def manual_download_instructions(self):
        return "just a sample data."

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate = 16000)
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        self.data_dir = abspath(expanduser(dl_manager.manual_dir))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": join(self.data_dir, "train.tsv"),
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding="utf-8") as f:
            data = f.read().strip()
            for id_, row in enumerate(data.split("\n")):
                path, sentence = tuple(row.split(" :: "))
                if exists(join(self.data_dir, path)):
                    with open(join(self.data_dir, path), 'rb') as audio_file:
                        audio_data = audio_file.read()
                    audio = {
                        "path": join(self.data_dir, path),
                        "bytes": audio_data,
                        "sampling_rate": 16_000
                    }
                    
                    yield id_, {
                        "file": join(self.data_dir, path),
                        "audio": audio,
                        "sentence": sentence,
                    }
                        
PERCENT_FILES = {
    '087797': '퍼센트',
    '215401': '퍼센트',
    '284574': '퍼센트',
    '397184': '퍼센트',
    '501006': '프로',
    '502173': '프로',
    '542363': '프로',
    '581483': '퍼센트'
}

def bracket_filter(sentence, mode='phonetic'):
  new_sentence = str()

  if mode == 'phonetic':
    flag = False

    for ch in sentence:
      if ch == '(' and flag is False:
        flag = True
        continue
      if ch == '(' and flag is True:
        flag = False
        continue
      if ch != ')' and flag is False:
        new_sentence += ch

  elif mode == 'spelling':
    flag = True

    for ch in sentence:
      if ch == '(':
        continue
      if ch == ')':
        if flag is True:
          flag = False
          continue
        else:
          flag = True
          continue
      if ch != ')' and flag is True:
        new_sentence += ch

  else:
    raise ValueError("Unsupported mode : {0}".format(mode))

  return new_sentence


def special_filter(sentence, mode='phonetic', replace=None):
  SENTENCE_MARK = ['?', '!', '.']
  NOISE = ['o', 'n', 'u', 'b', 'l']
  EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']

  new_sentence = str()
  for idx, ch in enumerate(sentence):
    if ch not in SENTENCE_MARK:
      if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
        continue

    if ch == '#':
      new_sentence += '샾'

    elif ch == '%':
      if mode == 'phonetic':
        new_sentence += replace
      elif mode == 'spelling':
        new_sentence += '%'

    elif ch not in EXCEPT:
      new_sentence += ch

  pattern = re.compile(r'\s\s+')
  new_sentence = re.sub(pattern, ' ', new_sentence.strip())
  return new_sentence


def sentence_filter(raw_sentence, mode='phonetic', replace=None):
  return special_filter(bracket_filter(raw_sentence, mode), mode, replace)