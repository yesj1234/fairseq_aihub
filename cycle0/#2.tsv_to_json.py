# 이거 왜 만든거더라.. 기억이 안나버리네
import os
import csv
import json


def main(args):
    # 1. load split.tsv
    for root, _, files in os.walk(args.split_path):
        for file in files:
            split_name, ext = os.path.splitext(file)
            if ext == ".tsv":
                # 2. generate json from the split.tsv
                new_json = {}
                rows = []
                with open(os.path.join(root, file), "r+", encoding="utf-8") as f:
                    split = csv.reader(f, delimiter="\n")
                    for row in split:
                        print(row)
                        source_lang, target_lang = row[0].split(" :: ")
                        rows.append({f"{args.source_lang}": source_lang, f"{args.target_lang}": target_lang})
                new_json["translation"] = rows
                #3. dump split.json
                with open(f"{split_name}.json", "w+", encoding="utf-8") as js:
                    json.dump(new_json, js, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", help="[split].tsv 파일 루트 경로")
    parser.add_argument("--source_lang", help="원래 언어")
    parser.add_argument("--target_lang", help="번역 언어")
    args = parser.parse_args()
    main(args)
