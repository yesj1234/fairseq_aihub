import os
import librosa
import shutil
from tqdm import tqdm, trange
import argparse


def main(args):
    if not os.path.exists('dataset/asr/kr-en/data'):
        os.makedirs('dataset/asr/kr-en/data/dev/txt', exist_ok=True)
        os.makedirs('dataset/asr/kr-en/data/test/txt', exist_ok=True)
        os.makedirs('dataset/asr/kr-en/data/train/txt',exist_ok=True)
        os.makedirs('dataset/asr/kr-en/data/dev/wav',exist_ok=True)
        os.makedirs('dataset/asr/kr-en/data/test/wav',exist_ok=True)
        os.makedirs('dataset/asr/kr-en/data/train/wav',exist_ok=True)      
        os.makedirs('result', exist_ok=True)
    splits = ['train', 'eval_clean', 'dev']
    for split in splits:
        print(f'splitting split {split}')
        trn_name = split+'.trn'
        if split=="eval_clean":
            split="test"
            wav_source=os.path.join(args.data_root, "평가용_데이터")
        else:
            wav_source=os.path.join(args.data_root, "한국어_음성_분야")
        print(trn_name)
        file_path_list=[]
        kr_transcription_list=[]
        en_translation_list=[]
        with open(os.path.join(args.data_root, "전시문_통합_스크립트", "KsponSpeech_scripts", trn_name), "r", encoding="utf-8") as trn_ref:
            lines = trn_ref.readlines()
            for line in lines:
                file_path, kr_transcription= line.split(" :: ")
                wav_file_path = file_path.replace(".pcm", ".wav")
                file_path_list.append(wav_file_path)
                kr_transcription_list.append(kr_transcription)
                en_translation_list.append("Filling english translation")
        print(f"file_path_list length: {len(file_path_list)}")
        print(f"kr_transcription_list length: {len(kr_transcription_list)}")
        print(f"en_translation_list length: {len(en_translation_list)}")
    
        g = open("dataset/asr/kr-en/data/"+split+"/txt/"+split+".kr", "a", encoding="utf-8")
        h = open("dataset/asr/kr-en/data/"+split+"/txt/"+split+".en", "a", encoding='utf-8')
        m = open("dataset/asr/kr-en/data/"+split+"/txt/"+split+".yaml", "a", encoding="utf-8")    
        if split == "test":
            for l in tqdm(trange(len(file_path_list)), desc="Preprocessing.."):
                # print(l, end="\r")
                wav_file = wav_source + '/' +file_path_list[l]
                destinaion = 'dataset/asr/kr-en/data/'+split+'/wav/'+file_path_list[l].split("/")[-1]
                shutil.copyfile(wav_file, destinaion)
                g.write(kr_transcription_list[l])
                h.write(en_translation_list[l]+'\n')
                m.write('- {duration: '+str(librosa.get_duration(path=wav_file))[:8]+', offset: 0.000000, speaker_id: spk.1, wav: '+file_path_list[l].split("/")[-1]+'}\n')
        else:
            for l in tqdm(trange(len(file_path_list)), desc="Preprocessing.."):
                # print(l, end="\r")
                wav_file = wav_source + '/' + file_path_list[l].split('/')[0] + '/' +file_path_list[l]
                destinaion = 'dataset/asr/kr-en/data/'+split+'/wav/'+file_path_list[l].split("/")[-1]
                shutil.copyfile(wav_file, destinaion)
                g.write(kr_transcription_list[l])
                h.write(en_translation_list[l]+'\n')
                m.write('- {duration: '+str(librosa.get_duration(path=wav_file))[:8]+', offset: 0.000000, speaker_id: spk.1, wav: '+file_path_list[l].split("/")[-1]+'}\n')
    g.close()
    h.close()
    m.close()
   

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="데이터가 저장된 폴더의 경로를 절대 경로로 입력")
    
    args = parser.parse_args()
    main(args)
