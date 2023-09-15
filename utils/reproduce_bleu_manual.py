from transformers import MBartTokenizer, MBartForConditionalGeneration
from evaluate import load
import os 
import argparse
import logging 


def main(args):
    logger = logging.getLogger("BLEU_GENERATOR")
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format='%(asctime)s-%(message)s',
        datefmt='%H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename='blue.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s|%(name)s|%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    
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
        logger.info(f"Prediction: {decoded_pred}")
        logger.info(f"Reference : {decoded_ref}")

        
    result = metric.compute(predictions=output_predictions, references=output_references)
    logger.info(f"result: {result}")    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--model_repo", required=True)
    parser.add_argument("--device", default="cpu", help="cuda if gpu is available")
    parser.add_argument("--src_lang", required=True, default="ko_KR")
    parser.add_argument("--tgt_lang", required=True, default="en_XX")
    # parser.add_argument("--log_file", required=True)
    args = parser.parse_args()
    main(args)


