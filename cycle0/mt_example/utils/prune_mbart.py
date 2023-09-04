import argparse
import os
from typing import List 

import torch #pip install torch 
from fairseq.data import Dictionary #pip install fairseq 

from transformers import (
	MBartForConditionalGeneration, MBartConfig, MBartTokenizer
)

def load_dict(langs: List[str], path: str) -> Dictionary:
	dict = Dictionary.load(path)
	for lang_code in langs:
		dict.add_symbol(f"[{lang_code}]")
	dict.add_symbol("<mask>")
	return dict

def main(args)->None:
	langs = args.langs.split(",")
	pre_dict = load_dict(langs, path=os.path.join(args.pre_dict))
	ft_dict = load_dict(langs, path=os.path.join(args.ft_dict))
	# data = torch.load(os.path.join(args.pre_train_dir, "model.pt"))
	model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model_from_hub)
	org_sd = model.state_dict()	
	resized_sd = model.state_dict()
 
 
	mapping:List[int] = []
	for i in range(len(ft_dict)):
		word = ft_dict[i]
		mapping.append(pre_dict.index(word))
	for name in ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "model.shared.weight", "lm_head.weight"]:
		pre_tensor:torch.Tensor = org_sd[name]
		ft_tensor = torch.zeros([len(ft_dict), 1024], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device)  
		for ft_i, pre_i in enumerate(mapping):
			ft_tensor[ft_i] = pre_tensor[pre_i]
		resized_sd[name] = ft_tensor
	resized_sd["final_logits_bias"] = resized_sd["final_logits_bias"][:, :len(ft_dict)]
	
	config = MBartConfig.from_pretrained(args.pretrained_model_from_hub)
	config.vocab_size = len(ft_dict)
	print(config)
	new_model = MBartForConditionalGeneration.from_pretrained(None, config = config, state_dict = resized_sd)
	new_model.save_pretrained(f"{args.output}")
	tokenizer = MBartTokenizer.from_pretrained(args.pretrained_model_from_hub)
	tokenizer.save_pretrained(args.output)
 
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--pre-dict", help="The pre-trained mBART model directory.")
	parser.add_argument("--ft-dict", help="The fine-tuning model dictionary.")
	parser.add_argument("--pretrained_model_from_hub", default="facebook/mbart-large-cc25", help="model name from huggingface hub")
	parser.add_argument("--langs", help="The pre-trained model languages.")
	parser.add_argument("--output", help="The trimmed mBART model.")
	args = parser.parse_args()
	main(args)
