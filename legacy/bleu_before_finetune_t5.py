from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import argparse

def get_data(domain):
    path = 'E:/Code/NLP/data/PRETRAIN_WMT14/json/'
    file_path = path + '{}.json'.format(domain)
    dataset = load_dataset('json', data_files=file_path)['train']
    loader = DataLoader(dataset, batch_size=args().batchsz)
    return loader

def compute_bleu(y_pred, y_true):
    metric = load_metric('bleu')
    metric.add_batch(predictions=y_pred, references=y_true)
    report = metric.compute()
    bleu = report['bleu'] * 100
    return bleu

def Model_Tokenizer(device):
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    return model.to(device), tokenizer

def evaluation(loader, model, tokenizer, device):
    y_true = []
    y_pred = []
    for i, batch in enumerate(loader):

        # Prepare and tokenize the source sentences
        src_sentences = [prefix + line for line in batch[args().src_language]]
        encoded_input = tokenizer(src_sentences, max_length=128,
                                  padding=True, truncation=True,
                                  return_tensors='pt', add_special_tokens=True).input_ids.to(device)

        # Translate and decode the inputs
        outputs = model.generate(encoded_input, max_length=175)
        batch_pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Concatenate the translated and reference sentences
        for sentence in batch[args().tgt_language]:
            sentence = tokenizer.tokenize(sentence)
            # print(sentence)
            y_true.append([sentence])
        for sentence in batch_pred:
            sentence = tokenizer.tokenize(sentence)
            # print(sentence)
            y_pred.append(sentence)

    bleu = compute_bleu(y_pred, y_true)
    print('Bleu Score: {:.2f}'.format(bleu))

def args():
    main_arg_parser = argparse.ArgumentParser(description="parser")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--gpu", type=int, default=0,
                                  help="assign gpu index")
    train_arg_parser.add_argument("--batchsz", type=int, default=128,
                                  help="batch size")
    train_arg_parser.add_argument("--model_name", type=str, default='t5-small',
                                  help="TBD")
    train_arg_parser.add_argument("--src_language", type=str, default='English',
                                  help="source language English")
    train_arg_parser.add_argument("--tgt_language", type=str, default='German',
                                  help="target language German")
    train_arg_parser.add_argument("--domain", type=str, default='WMT14_newstest2014_TEST',
                                  help="domain")
    return train_arg_parser.parse_args()

if __name__ == '__main__':

    device = torch.device('cuda:{}'.format(args().gpu)
                          if torch.cuda.is_available() else 'cpu')
    prefix = "Translate English to German: "

    print('--------------------------------- Using Device: {}'.format(device))


    print('Evaluating Domain: {}'.format(args().domain))
    loader = get_data(args().domain)
    model, tokenizer = Model_Tokenizer(device)
    evaluation(loader, model, tokenizer, device)