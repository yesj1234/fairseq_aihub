import os 
from pathlib import Path
import shutil
import json 
from tqdm import trange
import numpy as np
import librosa

def make_dirs(source='kr', target='en'):
    if not os.path.exists(f'dataset/{source}-{target}/data'):
        os.makedirs(f'dataset/{source}-{target}/data/train/txt', exist_ok=True)
        os.makedirs(f'dataset/{source}-{target}/data/train/wav', exist_ok=True)
        os.makedirs(f'dataset/{source}-{target}/data/test/txt', exist_ok=True)
        os.makedirs(f'dataset/{source}-{target}/data/test/wav', exist_ok=True)
        os.makedirs(f'dataset/{source}-{target}/data/dev/txt', exist_ok=True)
        os.makedirs(f'dataset/{source}-{target}/data/dev/wav', exist_ok=True)


def _load_json(path):
    with open(path, "r", encoding='utf-8') as json_ref:
        json_data = json.load(json_ref)
    return json_data 

def _write_file(m, h, g, split_translation, split_transcription, split_wav, wav_destination):
    for l in trange(len(split_wav)):
        file = split_wav[l]
        shutil.copy2(file,wav_destination)
        m.write(split_translation[l]+"\n")
        h.write(split_transcription[l]+"\n")
        g.write('- {duration: '+str(librosa.get_duration(filename=file))[:8]+', offset: 0.000000, speaker_id: spk.1, wav: '+file.split('/')[-1]+'}\n')


def main():
    source = 'kr'
    langs = ['en', 'de', 'it', 'fr']
    os.makedirs('results', exist_ok=True)
    lang_dict = {'en': '영국영어',
                 'de': '독일어', 
                 'it': '이태리어',
                 'fr': '프랑스어'
                }
    for lang in langs:
        make_dirs(source=source, target=lang)
        metadata_root=Path('C:\\Users\\must_\\Downloads\\ko_eu_sample\\03.샘플데이터\\2.라벨링데이터\\통번역'+ "\\"+ lang_dict[lang])
        
        translation_list = []
        transcription_list = []
        wav_list = []
        splits = ["train", "dev", "test"]

        for root, dir, files in os.walk(metadata_root):
            if files:
                for file in files:
                    json_data = _load_json(os.path.join(root, file))
                    wav_file, _ = os.path.splitext(json_data["S-FName"])
                    wav_folder = "_".join(wav_file.split("_")[:4])
                    wav_root = 'C:\\Users\\must_\\Downloads\\ko_eu_sample\\03.샘플데이터\\1.원천데이터\\한국어' + "\\" + json_data["대분류"].replace("/", "") + "\\" + wav_folder
                    wav_path = wav_root + "\\" + json_data["S-FName"]
                    wav_list.append(wav_path)
                    transcription_list.append(json_data["원문"])
                    translation_list.append(json_data["최종번역문"])
        
        train_transcription, dev_transcription, test_transcription = np.split(transcription_list, [int(len(wav_list) * 0.9), int(len(wav_list) * 0.98)])
        train_translation, dev_translation, test_translation = np.split(translation_list,[int(len(wav_list) * 0.9), int(len(wav_list) * 0.98)])
        train_wav, dev_wav, test_wav = np.split(wav_list,[int(len(wav_list) * 0.9), int(len(wav_list) * 0.98)])

        for split in splits:
            destination = Path(f'{os.getcwd()}/dataset/{source}-{lang}/data/{split}/txt')
            wav_destination = Path(f'{os.getcwd()}/dataset/{source}-{lang}/data/{split}/wav')
            m = open(os.path.join(destination, f'{split}.{lang}') , "w", encoding='utf-8') #translation 
            h = open(os.path.join(destination, f'{split}.{source}') , "w", encoding='utf-8') #transcription
            g = open(os.path.join(destination, f'{split}.yaml') , "w", encoding='utf-8') 
            if split == "train":
                _write_file(m, h, g, train_translation,train_transcription,  train_wav, wav_destination)
            elif split == "dev":
                _write_file(m, h, g, dev_translation, dev_transcription, dev_wav, wav_destination)
            else:
                _write_file(m, h, g, test_translation, test_transcription, test_wav, wav_destination)
            m.close()
            h.close()
            g.close()
    



if __name__ == "__main__":
    main()