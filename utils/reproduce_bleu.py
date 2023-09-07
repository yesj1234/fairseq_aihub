from transformers import MBartTokenizer, MBartForConditionalGeneration
from evaluate import load
import os 
import torch 

inputs = []
references = []
with open(os.path.join('/home/work/cycle0/한국어_영어/mt_split/test.tsv'), "r+", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        test_input, reference = line.split(" :: ") 
        inputs.append(test_input)
        references.append(reference)
        
model = MBartForConditionalGeneration.from_pretrained("yesj1234/mbart_cycle0_ko-en")
model.eval()
tokenizer = MBartTokenizer.from_pretrained("yesj1234/mbart_cycle0_ko-en")
batches = tokenizer.prepare_seq2seq_batch(src_texts = inputs[:5], src_lang="ko_KR", tgt_lang="en_XX", tgt_texts=references[:5])
pt_inputs = torch.LongTensor(batches["input_ids"])


output = model.generate(inputs = pt_inputs)
#next, decode the generated output

#next, tokenize the reference sentences 
#next, call metric.compute with generated decoded prediction and decoded reference.






print(output)





# predictions = model.predict()

# metric = load("sacrebleu")

# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = [[label.strip()] for label in labels]
#     return preds, labels







