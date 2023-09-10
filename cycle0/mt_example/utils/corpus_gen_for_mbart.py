# Create train.src_lang & train.tgt_lang From train.tsv
# Create test.src_lang & test.tgt_lang From test.tsv 
# Create validation.src_lang & test.tgt_lang From validation.tsv 

import os 
import argparse 
import csv 

def main(args):
    for root, _dirs, files in os.walk(args.splits):
        if files:
            for file in files:
                file_name, ext = os.path.splitext(file)
                if ext == ".tsv":
                    split = file_name
                    with open(os.path.join(root, file), "r+", encoding="utf-8") as split_file:
                        tsv_reader = csv.reader(split_file, delimiter = '\t')
                        src_lines, tgt_lines = [], []
                        for row in tsv_reader:
                            if row:
                                src_line, tgt_line = row[0].split(" :: ")    
                                src_lines.append(src_line)
                                tgt_lines.append(tgt_line)
                        with open(os.path.join(root, f"{split}.{args.source_lang}"), "w+", encoding="utf-8") as src_new, open(os.path.join(root, f"{split}.{args.target_lang}"), "w+", encoding="utf-8") as tgt_new:
                            for i in range(len(src_lines)):
                                src_new.write(f"{src_lines[i]}\n")
                                tgt_new.write(f"{tgt_lines[i]}\n")
                                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", help="path for splits directory containing train.tsv, test.tsv, validation.tsv")
    parser.add_argument("--source_lang", help="ko", default="ko", required=True)
    parser.add_argument("--target_lang", help="en ja zh", required=True)
    args = parser.parse_args()
    main(args)
    
    