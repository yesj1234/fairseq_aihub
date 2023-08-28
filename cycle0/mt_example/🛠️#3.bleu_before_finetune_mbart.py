from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import load_dataset 
from evaluate import load 
import csv

def get_model_and_tokenizer():
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-cc25") 
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    return tokenizer, model 

def get_metric(metric):
    return load(metric)

# def main(args):
#     #1. load the test dataset from test.json 
#     test_split = load_dataset('json', data_files = args.test_dir)
#     print(test_split)
#     #2. load the tokenizer, model, and metric and set lang code for tokenizer. 
#     tokenizer, model = get_model_and_tokenizer()
#     tokenizer.src_lang = args.source_lang
#     tokenizer.tgt_lang = args.target_lang 
    
#     metric = get_metric(args.metric)
    
#     #3. batch decode 
#     references = []
#     predictions = []
#     #4. compute the metric 
#     metric.compute(references = references, predictions = predictions) 
 

def main(args):
    #1. load the test dataset from test.json 
    #2. load the tokenizer, model, and metric and set lang code for tokenizer. 
    tokenizer, model = get_model_and_tokenizer()
    tokenizer.src_lang = args.source_lang
    tokenizer.tgt_lang = args.target_lang 
    
    metric = get_metric(args.metric)
    
    #3. batch decode
    sources = [] 
    references = []
    predictions = []
    with open(args.test_dir, "r+", encoding="utf-8") as f:
        split = csv.reader(f, delimiter="\t")
        for row in split:
            source, target = row[0].split(" :: ")
            sources.append(source)
            references.append(target)
            
    #4. compute the metric 
    metric.compute(references = references, predictions = predictions)  
 
    
    
    
if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_lang", help="language code for mbart : Korean (ko_KR), Japanese (ja_XX), English (en_XX), Chinese (zh_CN)")
    parser.add_argument("--target_lang", help="language code for mbart : Korean (ko_KR), Japanese (ja_XX), English (en_XX), Chinese (zh_CN)")
    parser.add_argument("--test-dir", help="path to the test.json")
    parser.add_argument("--metric", help="Metric for mt. BLEU by default", default="bleu")
    args = parser.parse_args()
    main(args)

