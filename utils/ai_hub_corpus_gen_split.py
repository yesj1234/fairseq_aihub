import os
import argparse
import pandas as pd



def main(args):
    #1. loop through the data root
    source  = args.data_root
    columns = ["원문", "번역문"]
    corpus_df = pd.DataFrame(columns=columns)
    for i, file in enumerate(os.listdir(source)):
        ext = os.path.splitext(file)[1]
        if ext == ".xlsx":
            df = pd.read_excel(file)
            df = df.loc[:, columns]
            corpus_df = pd.concat([corpus_df, df], ignore_index=True)

    #Split Dataset
    # test : 3000 
    # val : 5000
    # train : 1493750 
    splits = ["test", "val", "train"]
    split_point = [3000, 8000, len(corpus_df)]
    #shuffle the data 
    corpus_df = corpus_df.sample(frac=1).reset_index(drop=True)
    splits =["test", "val", "train"]

    for i in range(len(splits)):
        source_file_name = f"{splits[i]}.ko"
        target_file_name = f"{splits[i]}.en"
        
        if source_file_name == "test.ko":
            corpus_df.loc[:3000, columns[0]].to_csv(source_file_name)
            corpus_df.loc[:3000, columns[1]].to_csv(target_file_name)
        elif source_file_name == "val.ko":
            corpus_df.loc[3000:8000, columns[0]].to_csv(source_file_name)
            corpus_df.loc[3000:8000, columns[1]].to_csv(target_file_name)
        else:
            corpus_df.loc[8000:, columns[0]].to_csv(source_file_name)
            corpus_df.loc[8000:, columns[1]].to_csv(target_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=str)
    args = parser.parse_args()
    
    main(args)





