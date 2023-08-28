import os
import sentencepiece as spm
from .utils import build
from .utils import spm_encode 
from .utils import prune_mbart
# from typing import List


def main(args):
    # 1. Create a corpus for source lang to target lang.
    # for example, train.ko train.en / test.ko test.en / valid.ko valid.en
    for path in [args.train_set, args.test_set, args.valid_set]:
        cur_split = path.split("/")[-1].split(".")[-1]
        source, target = [], []
        with open(path, "r+", encoding="utf-8") as split_file:
            lines = split_file.readlines()
            for line in lines:
                src, tgt = line.split(" :: ")
                source.append(src)
                target.append(tgt)
        source_dest = os.path.join(
            args.output_dir, f"{cur_split}.{args.source_lang}")
        target_dest = os.path.join(
            args.output_dir, f"{cur_split}.{args.target_lang}")
        with open(source_dest, "w+") as src_file, open(target_dest, "w+") as tgt_file:
            for i in range(len(source)):
                src_file.write(f"{source[i]}\n")
                tgt_file.write(f"{target[i]}\n")
            # 2. update all the sentence pairs of corpus with the tokenized version with sentencepiece.
            # for example train.spm.ko train.spm.en / test.spm.ko test.spm.en / valid.spm.ko valid.spm.en
            spm_encode(f"--model {args.spm_model} --inputs {src_file} --outputs {tgt_file}")
    #3. run build.py to generate a new dict.txt
    build(f"--corpus-data {os.path.join(args.output_dir)}/*.spm.* \
        --langs {args.langs} \
            --output {os.path.join(args.output_dir)}/dict.txt")
    #4. go for pruning pretrained model by using prune_mbart.py to generate lighter version of pretrained model for source_lang - target_lang language pair
    ft_dict = os.path.join(args.output_dir, "dict.txt")
    prune_mbart(f"--pre-train-dir {args.pre_train_dir} \
        --ft-dict {ft_dict} \
        --langs {args.langs} \
        --output {args.output_dir}")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # argugments for step1 
    parser.add_argument(
        "--train-set", help="file path for train.tsv file", required=True)
    parser.add_argument(
        "--test-set", help="file path for test.tsv file", required=True)
    parser.add_argument(
        "--valid-set", help="file path for valid.tsv file", required=True)
    parser.add_argument(
        "--source-lang", help="source language. ko/en/ja/ch", required=True)
    parser.add_argument(
        "--target-lang", help="target language. ko/en/ja/ch", required=True)
    parser.add_argument(
        "--output-dir", help="output dir of the generated files", required=True)
    # arguments for step2
    parser.add_argument(
        "--spm-model", help="mbart.cc25 sentencepiece.bpe.model path", required=True)
    parser.add_argument("--langs", help="the pre-trained model languages")
    # arguments for step3
    parser.add_argument("--pre-train-dir", help="path to the pretrained mbart cc25 model")
    args = parser.parse_args()
    main(args)
