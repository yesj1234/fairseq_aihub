from transformers import AutoTokenizer, MBartForConditionalGeneration
import argparse
import evaluate

def main(args):    
    tokenizer = AutoTokenizer.from_pretrained(args.model_check_point)
    model = MBartForConditionalGeneration.from_pretrained(args.model_check_point)
    tokenizer.src_lang = args.src_lang 
    metric = evaluate.load(args.metric)
    def translate(text):
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(args.device)
        attention_mask = inputs.attention_mask.to(args.device)
        output = model.generate(input_ids, attention_mask=attention_mask, forced_bos_token_id = tokenizer.lang_code_to_id[args.target_lang])
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    
    test_sentences = []
    test_sentences_references = []
    with open(args.test_file, "r+", encoding="utf-8") as test_file:
        lines = test_file.readlines()
        for line in lines:
            source, reference = line.split(" :: ")
            test_sentences.append(source)
            test_sentences_references.append(reference.replace("\n", ""))
    test_predictions = list(map(translate, test_sentences))
    for i in range(len(test_sentences)):
        print(f"prediction: {test_predictions[i]}")
        print(f"reference: {test_sentences_references[i]}")
    results = metric.compute(predictions = test_predictions, references = test_sentences_references)
    print(results) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_check_point", help="huggingface model directory name")
    parser.add_argument("--metric", help="evaluation metric for the model. [bleu, sacrebleu]", default="sacrebleu")
    parser.add_argument("--src_lang", help="source lang for mbart generation. ko_KR, en_XX, ja_XX, zh_CN", default="ko_KR")
    parser.add_argument("--target_lang", help="target lang for mbart generation. ko_KR, en_XX, ja_XX, zh_CN")
    parser.add_argument("--device", default="cpu", help="cuda if gpu available else cpu by default")
    parser.add_argument("--test_file", help="tsv file path for evaluating the model BLEU")
    
    args = parser.parse_args()
    main(args)