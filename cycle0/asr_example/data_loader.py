import os
import datasets 
import re

from datasets.download.download_manager import DownloadManager

_DESCRIPTION = "sample data for Cycle 0"

class CleaveSpeech(datasets.GeneratorBasedBuilder): 
    """Returns SplitGenerators."""
    VERSION = datasets.Version("0.0.1")
    def _split_generators(self, dl_manager: DownloadManager):
        self.data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(self.data_dir, "asr_train.csv"),
                    "split": "train"
                }
            )
        ]    
    def _generate_examples(self, filepath, split): 
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding='utf-8') as f:
            data = f.read().strip()
            