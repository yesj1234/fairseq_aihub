import os
import datasets 
import re

from datasets.download.download_manager import DownloadManager

_DESCRIPTION = "sample data for Cycle 0"

class CleaveSpeech(datasets.GeneratorBasedBuilder): 
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000)
                }
            )
        )
    
    """Returns SplitGenerators."""
    VERSION = datasets.Version("0.0.1")
    def _split_generators(self, dl_manager: DownloadManager):
        self.data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(self.data_dir, "asr_train.tsv"),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(self.data_dir, "asr_test.tsv"),
                    "split": "test"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(self.data_dir, "asr_validation.tsv"),
                    "split": "validation"
                }
            )
        ]    
    def _generate_examples(self, filepath, split): 
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding='utf-8') as f:
            data = f.read().strip()
            for id_, row in enumerate(data.split("\n")):
                path, sentence = tuple(row.split(" :: "))
                # FIGURE OUT how to download and extract the audio in local file system.
                # 1. read the audio file. audio_data = audio_file.read()
                # 2. audio = {
                #       "path": path to the audio file,
                #       "bytes": audio_data,
                #       "sampling_rate": 16_000
                # }
                # 3. yield id_, {
                #    "file": path to the audio,
                #    "audio": audio,
                #    "sentence": sentence
                # }
                