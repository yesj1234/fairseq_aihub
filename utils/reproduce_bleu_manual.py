from transformers import MBartTokenizer, MBartForConditionalGeneration
from evaluate import load
import os 
import argparse


def main(args):
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels
    
    
    inputs = []
    references = []
    with open(os.path.join(args.test_split), "r+", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            test_input, reference = line.split(" :: ") 
            inputs.append(test_input.strip())
            references.append(reference.strip())

    model = MBartForConditionalGeneration.from_pretrained(args.model_repo)
    model.to(args.device)
    tokenizer = MBartTokenizer.from_pretrained(args.model_repo)
    tokenizer.src_lang = args.src_lang # in mBART lang code. ko_KR , ja_XX, zh_CN, en_XX 
    tokenizer.tgt_lang = args.tgt_lang # in mBART lang code. ko_KR, ja_XX, zh_CN, en_XX
    metric = load("sacrebleu")
    total_test_length = len(inputs)
    iter_inputs = iter(inputs)
    iter_references = iter(references)
    current_test_count = 0
    output_predictions = []
    output_references = []
    while current_test_count < total_test_length:
        current_test_count += 1
        pt_inputs = tokenizer(next(iter_inputs), return_tensors = "pt").to(args.device)
        encoded_reference = tokenizer(next(iter_references), return_tensors = "pt").to(args.device)
        # pred = model.generate(**pt_inputs, decoder_start_token_id = tokenizer.lang_code_to_id[args.tgt_lang])
        pred = model.generate(**pt_inputs, forced_bos_token_id = tokenizer.lang_code_to_id[args.tgt_lang], num_beams = 5)
        decoded_pred = tokenizer.decode(pred[0], skip_special_tokens=True)
        decoded_ref = tokenizer.decode(encoded_reference["input_ids"][0], skip_special_tokens=True)
        output_predictions.append(decoded_pred)
        output_references.append(decoded_ref)
        print(f"decoded_pred: {decoded_pred}")
        print(f"decoded_ref : {decoded_ref}")
        
    result = metric.compute(predictions=output_predictions, references=output_references)
    # result = metric.compute(predictions=output_predictions, references=references)

    print(result)        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--model_repo", required=True)
    parser.add_argument("--device", default="cpu", help="cuda if gpu is available")
    parser.add_argument("--src_lang", required=True, default="ko_KR")
    parser.add_argument("--tgt_lang", required=True, default="en_XX")
    args = parser.parse_args()
    main(args)

