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

_CITATION = """\
@article{bang2020ksponspeech,
  title={KsponSpeech: Korean spontaneous speech corpus for automatic speech recognition},
  author={Bang, Jeong-Uk and Yun, Seung and Kim, Seung-Hi and Choi, Mu-Yeol and Lee, Min-Kyu and Kim, Yeo-Jeong and Kim, Dong-Hyun and Park, Jun and Lee, Young-Jik and Kim, Sang-Hun},
  journal={Applied Sciences},
  volume={10},
  number={19},
  pages={6936},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
"""

_DESCRIPTION = """\
KsponSpeech is a large-scale spontaneous speech corpus of Korean conversations. This corpus contains 969 hrs of general open-domain dialog utterances, spoken by about 2,000 native Korean speakers in a clean environment. All data were constructed by recording the dialogue of two people freely conversing on a variety of topics and manually transcribing the utterances. The transcription provides a dual transcription consisting of orthography and pronunciation, and disfluency tags for spontaneity of speech, such as filler words, repeated words, and word fragments. KsponSpeech is publicly available on an open data hub site of the Korea government. (https://aihub.or.kr/aidata/105)
"""

_HOMEPAGE = "https://aihub.or.kr/aidata/105"


class KsponSpeech(datasets.GeneratorBasedBuilder):
    """The Korean Spontaneous Speech Corpus for Automatic Speech Recognition (KsponSpeech)"""

    VERSION = datasets.Version("0.1.0")

    @property
    def manual_download_instructions(self):
        return "To use KsponSpeech, data files must be downloaded manually to a local drive. Please submit your request on the official website (https://aihub.or.kr/aidata/105). Once your request is approved, download all files, extract .zip files in one folder, and load the dataset with `datasets.load_dataset('ksponspeech', data_dir='path/to/folder')`."

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        self.data_dir = abspath(expanduser(dl_manager.manual_dir))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": join(self.data_dir, "scripts/train.trn"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": {
                        "clean": join(self.data_dir, "scripts/eval_clean.trn"),
                        "other": join(self.data_dir, "scripts/eval_other.trn"),
                    },
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": join(self.data_dir, "scripts/dev.trn"),
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples as (key, example) tuples."""
        if split is "test":
            with open(filepath["clean"], encoding="utf-8") as f1, open(
                filepath["other"], encoding="utf-8"
            ) as f2:
                data = "\n".join([f1.read().strip(), f2.read().strip()])
                for id_, row in enumerate(data.split("\n")):
                    path, sentence = tuple(row.split(" :: "))
                    yield id_, {
                        "file": join(self.data_dir, path),
                        "sentence": sentence,
                    }
        else:
            with open(filepath, encoding="utf-8") as f:
                data = f.read().strip()
                for id_, row in enumerate(data.split("\n")):
                    path, sentence = tuple(row.split(" :: "))
                    yield id_, {
                        "file": join(self.data_dir, path),
                        "sentence": sentence,
                    }