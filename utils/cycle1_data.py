import datasets 
import os 

class Opus100(datasets.GeneratorBasedBuilder):        
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({"translation": datasets.features.Translation(languages=("ko", "ja"))})
        )
    def _split_generators(self, dl_manager):
        self.data_dir = os.environ["DATA_DIR"]
        
        return [
            datasets.SplitGenerator(
            name=datasets.Split.TEST,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                "filepath": self.data_dir,
            },
        )
        ]
        


    def _generate_examples(self, filepath):
        """Yields examples."""
        src_tag, tgt_tag = "ko", "ja"
        src, tgt = None, None
        with open(filepath, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                src, tgt = line.split(" :: ")
                src, tgt = src.strip(), tgt.strip()
                yield idx, {"translation": {src_tag: src, tgt_tag: tgt}}
            