from transformers import (
    Seq2SeqTrainer,
    AutoModelForSeq2SeqLM,
    MBartTokenizer,
    default_data_collator
)
from datasets import load_dataset
from evaluate import load
import numpy as np 

def main(args):

    data = load_dataset("./cycle1_data.py")
    # print(data)
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels
    
    metric = load("sacrebleu")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        print(f"decoded_preds : {decoded_preds}")
        print(f"decoded_labels: {decoded_labels}")
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result  
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args.repo_id)
    tokenizer = MBartTokenizer.from_pretrained(args.repo_id)
    trainer = Seq2SeqTrainer(model=model, 
                             eval_dataset=data["test"], 
                             tokenizer = tokenizer,
                             compute_metrics = compute_metrics,
                             data_collator=default_data_collator)
    res = trainer.evaluate(max_length=1024, num_beams=1, metric_key_prefix="eval")
    print(res)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="yesj1234/mbart_cycle1_ko-ja")
    args = parser.parse_args()
    main(args)