from datasets import load_dataset
from evaluate import load 
import librosa
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoModelFor   

if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    LANG_ID = "en"
    MODEL_CHECKPOINT = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    

    test_dataset = load_dataset("common_voice", LANG_ID, split=f"test")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.to(device)
    
    # Preprocessing the datasets.
    # We need to read the audio files as arrays
    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
        batch["speech"] = speech_array
        batch["sentence"] = batch["sentence"].upper()
        return batch

    test_dataset = test_dataset.map(speech_file_to_array_fn)
    inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)
    y_pred = []
    y_true = []

    for i, predicted_sentence in enumerate(predicted_sentences):
        print("-" * 100)
        print("Reference:", test_dataset[i]["sentence"])
        print("Prediction:", predicted_sentence)
        y_pred.append(predicted_sentence)
        y_true.append(test_dataset[i]["sentence"])


    cer = load("cer")
    cer.compute(predictions= y_pred, references=y_true)
    print(cer)

