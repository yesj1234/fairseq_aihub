# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition."""


import os
import re
from pathlib import Path

import datasets
import numpy as np
import librosa
from datasets.tasks import AutomaticSpeechRecognition


_CITATION = """\
@Article{app10196936,
AUTHOR = {Bang, Jeong-Uk and Yun, Seung and Kim, Seung-Hi and Choi, Mu-Yeol and Lee, Min-Kyu and Kim, Yeo-Jeong and Kim, Dong-Hyun and Park, Jun and Lee, Young-Jik and Kim, Sang-Hun},
TITLE = {KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition},
JOURNAL = {Applied Sciences},
VOLUME = {10},
YEAR = {2020},
NUMBER = {19},
ARTICLE-NUMBER = {6936},
URL = {https://www.mdpi.com/2076-3417/10/19/6936},
ISSN = {2076-3417},
DOI = {10.3390/app10196936}
}
"""

_DESCRIPTION = """\
This paper introduces a large-scale spontaneous speech corpus of Korean, named KsponSpeech. This corpus contains 969 h of general open-domain dialog utterances, spoken by about 2000 native Korean speakers in a clean environment. All data were constructed by recording the dialogue of two people freely conversing on a variety of topics and manually transcribing the utterances. The transcription provides a dual transcription consisting of orthography and pronunciation, and disfluency tags for spontaneity of speech, such as filler words, repeated words, and word fragments. This paper also presents the baseline performance of an end-to-end speech recognition model trained with KsponSpeech. In addition, we investigated the performance of standard end-to-end architectures and the number of sub-word units suitable for Korean. We investigated issues that should be considered in spontaneous speech recognition in Korean. KsponSpeech is publicly available on an open data hub site of the Korea government.
More info on KsponSpeech dataset can be understood from the webpage which can be found here:
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123
"""

_HOMEPAGE = "https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123"

_ROOT_DIRNAME = "ksponspeech"
_SCRIPT_DIRNAME = "KsponSpeech_scripts"

_SCRIPT_SPLITS = {
  "train": "train.trn",
  "dev": "dev.trn",
  "eval_clean": "eval_clean.trn",
  "eval_other": "eval_other.trn"
}

class KsponSpeechConfig(datasets.BuilderConfig):
  """BuilderConfig for KsponSpeech."""

  def __init__(self, **kwargs):
    """
    Args:
      data_dir: `string`, the path to the folder containing the files in the
        downloaded .tar
      citation: `string`, citation for the data set
      url: `string`, url for information about the data set
      **kwargs: keyword arguments forwarded to super.
    """
    # version history
    # 0.1.0: First release
    super(KsponSpeechConfig, self).__init__(version=datasets.Version("0.1.0", ""), **kwargs)
      


class KsponSpeech(datasets.GeneratorBasedBuilder):
  """KsponSpeech dataset."""

  @property
  def manual_download_instructions(self):
    return (
      "To use KsponSpeech you have to download it manually. "
      "Please create an account and download the dataset from "
      "https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=123 \n"
      "Then load the dataset with: "
      "`datasets.load_dataset('ksponspeech', data_dir='path/to/folder/folder_name')`"
    )

  def _info(self):
    return datasets.DatasetInfo(
      description=_DESCRIPTION,
      features=datasets.Features(
        {
          "id": datasets.Value("string"),
          "audio": datasets.Audio(sampling_rate=16_000),
          "text": datasets.Value("string")
        }
      ),
      supervised_keys=("file", "text"),
      homepage=_HOMEPAGE,
      citation=_CITATION,
      task_templates=[AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")],
    )

  def _split_generators(self, dl_manager):
    # Step 1. Extract all zip files
    # Step 2. Get scripts
    # Step 3. Generate samples
    data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
    data_dir = os.path.join(data_dir, _ROOT_DIRNAME)
    if not os.path.exists(data_dir):
      raise FileNotFoundError(
        f"{data_dir} does not exist. Make sure you insert a manual dir via"
        "`datasets.load_dataset('ksponspeech', data_dir=...)`"
        "that includes files. Manual download instructions:"
        f"{self.manual_download_instructions}"
      )
    archive_paths = {}
    for fname in os.listdir(data_dir):
      if not '.lock' in fname:
        fname_no_ext = os.path.splitext(fname)[0]
        archive_paths[fname_no_ext] = os.path.join(data_dir, fname)
    local_extracted_archives = dl_manager.extract(archive_paths)
    script_archive_path = local_extracted_archives[_SCRIPT_DIRNAME]
    return [
      datasets.SplitGenerator(
        name=datasets.Split.TRAIN, 
        gen_kwargs={
          "script": os.path.join(script_archive_path, _SCRIPT_SPLITS['train']),
          "local_extracted_archives": local_extracted_archives
        }
      ),
      datasets.SplitGenerator(
        name=datasets.Split.VALIDATION, 
        gen_kwargs={
          "script": os.path.join(script_archive_path, _SCRIPT_SPLITS['dev']),
          "local_extracted_archives": local_extracted_archives
        }
      ),
      datasets.SplitGenerator(
        name="eval.clean", 
        gen_kwargs={
          "script": os.path.join(script_archive_path, _SCRIPT_SPLITS['eval_clean']),
          "local_extracted_archives": local_extracted_archives
        }
      ),
      datasets.SplitGenerator(
        name="eval.other", 
        gen_kwargs={
          "script": os.path.join(script_archive_path, _SCRIPT_SPLITS['eval_other']),
          "local_extracted_archives": local_extracted_archives
        }
      ),
    ]

  def _generate_examples(self, script, local_extracted_archives):
    """Generate examples from KsponSpeech archive_path based on the test/train trn information."""
    # Iterating the contents of the data to extract the relevant information
    with open(script) as f:
      for key, line in enumerate(f):
        audio_path, text = line.split(' :: ')
        audio_subdir = audio_path.split('/')[0]
        if os.path.basename(audio_path)[12:18] in PERCENT_FILES.keys():
            replace = PERCENT_FILES[os.path.basename(audio_path)[12:18]]
        else:
            replace = None
        text = sentence_filter(text, replace=replace).strip()
        if 'KsponSpeech_eval/' in audio_path:
          audio_path = audio_path.replace('KsponSpeech_eval/','')
        audio_path = os.path.join(local_extracted_archives[audio_subdir], audio_path)
        if os.path.exists(audio_path):
          with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read() 
            if len(audio_data) % 2 != 0:
              # Remove unknown additional bytes in KspoonSpeech_eval
              audio_data = audio_data[:-1]
          audio = {
            "path": audio_path,
            "bytes": audio_data,
            "sampling_rate": 16_000
          }
          yield key, {
            "id": os.path.splitext(os.path.basename(audio_path))[0],
            "audio": audio,
            "text": text
          }

# ------------------------------------------------------------------------
# following codes are copied from https://github.com/sooftware/ksponspeech

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