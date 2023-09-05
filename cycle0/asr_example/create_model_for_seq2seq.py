#!/usr/bin/env python3
from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
import torch


encoder_id = "facebook/wav2vec2-base"
decoder_id = "facebook/bart-base"

model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id, encoder_add_adapter=True)
model.config.encoder.feat_proj_dropout = 0.0
model.config.encoder.mask_time_prob = 0.0
model.config.decoder_start_token_id = model.decoder.config.bos_token_id
model.config.pad_token_id = model.decoder.config.pad_token_id
model.config.eos_token_id = model.decoder.config.eos_token_id
model.config.max_length = 40
model.config.encoder.layerdrop = 0.0
model.config.use_cache = False
model.config.processor_class = "Wav2Vec2Processor"

# check if generation works
out = model.generate(torch.ones((1, 2000)))

model.save_pretrained("./seq2seq_cycle1_ko")

feature_etxractor = AutoFeatureExtractor.from_pretrained(encoder_id)
feature_etxractor.save_pretrained("./seq2seq_cycle1_ko")
tokenizer = AutoTokenizer.from_pretrained(decoder_id)
tokenizer.save_pretrained("./seq2seq_cycle1_ko")