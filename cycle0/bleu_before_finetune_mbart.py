from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import os
import argparse
from evaluate import load
import torch

def get_model_and_tokenizer(device):
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    return model.to(device), tokenizer 

def compute_bleu(y_pred, y_true):
    metric = load("bleu")
    result = metric.compute(references=y_true, predictions=y_pred)
    bleu = result["bleu"] * 100 
    return bleu


def main(args):
    model, tokenizer = get_model_and_tokenizer()
    tokenizer.src_lang = args.src_lang

    article_ja = []
    article_en = []
    with open(os.path.join(args.data_dir, "dev"), "r", encoding="utf-8") as test_file:
        for line in test_file: 
            en, jp = line.split("\t")
            article_ja.append(jp)
            article_en.append(en)
    
    encoded_ja = tokenizer(article_ja, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = model.generate(
        **encoded_ja,
        forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang],
        max_length=128
    )
    preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, batch_size=4)

    bleu = compute_bleu(y_pred = preds, y_true = article_en)
    print(f"Bleu Score: {bleu:.2f }")
    
    
if __name__ == "__main__":
    import time
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help="data dir path of splits. train/test/dev")
    parser.add_argument("--src_lang", type=str, help="ja_XX: japanese, zh_CN: chinese, en_XX: english, ko_KR: korean")
    parser.add_argument("--tgt_lang", type=str, help="ja_XX: japanese, zh_CN: chinese, en_XX: english, ko_KR: korean")
    parser.add_argument("--gpu", type=int, default=0, help="assign gpu index")
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args().gpu)
                          if torch.cuda.is_available() else 'cpu')
    main(args)
    end = time.time()
    print("-"*10 + f"{end-start} seconds" + "-"*10)