########################################################################################################################################################################
# {     
#     "fi_sound_filename": "54_106_11.60_19.34.wav",
#     "fi_sound_filepath": "https://objectstorage.ap-seoul-1.oraclecloud.com/n/cnb97trxvnun/b/clive-resource/o/output/한국어_영어/원천데이터/교육/54/54_106_11.60_19.34.wav",
#     "tc_text": "네  유치원 교육과정 A형 어 저희가 보도록 하겠습니다. 2013학년도 유치원 교육과정 A입니다.",
#     "tl_trans_lang": "영어",
#     "tl_trans_text": "We'll take a look at Kindergarten Curriculum Type A. This is kindergarten Curriculum A for the 2013 school year.",
# }
##########################################################################################################################################################################
# 카테고리 코드정리
# 일상,소통 : ca1
# 여행 : ca2
# 게임 : ca3
# 경제 : ca4
# 교육 : ca5
# 스포츠 : ca6
# 라이브커머스 : ca7
# 음식,요리 : ca8
# 운동,건강 : ca9
# 패션,뷰티 : ca10
# 예시 교육 -> 교육_ca5
# 폴더 구조 변경
# 한국어 -> (KO)
# 영어 -> (EN)
# 일본어 -> (JP)
# 중국어 -> (CH)

import json 
import os
import argparse
import numpy as np
from typing import Dict

CATEGORY:Dict[str, str] = {
    "일상/소통_ca1" : "일상/소통",
    "여행_ca2" : "여행",
    "게임_ca3" : "게임",
    "경제_ca4" : "경제",
    "교육_ca5" : "교육",
    "스포츠_ca6" : "스포츠",
    "라이브커머스_ca7" : "라이브커머스",
    "음식/요리_ca8" : "음식/요리",
    "운동/건강_ca9" : '운동/건강',
    "패션/뷰티_ca10" : "패션/뷰티",
    "한국어(KO)" : "한국어" ,
    "영어(EN)" : "영어",
    "일본어(JP)" : "일본어",
    "중국어(CH)" : "중국어"
}

def get_neccesary_info(json_file):
    def _replace_path(path):
        for key in CATEGORY:
            path = path.replace(CATEGORY[key], key)
        return path
    json_data = json.load(json_file)
    path = json_data["fi_sound_filepath"].split("/")[-5:]
    path = '/'.join(path)
    path = _replace_path(path)
    transcription = json_data["tc_text"]
    return path, transcription


def main(args):
    os.makedirs(os.path.join(args.asr_dest_folder, "asr_split"), exist_ok=True)

    sound_file_paths, sound_file_transcriptions = [], []
    for root, dir, files in os.walk(args.jsons):
        if files:
            print(f"json files from {os.path.join(root)}")
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == ".json":
                    with open(os.path.join(root, file), "r", encoding="utf-8") as json_file:
                        path, transcription = get_neccesary_info(json_file)
                        sound_file_paths.append(path)
                        sound_file_transcriptions.append(transcription)
                        
    sound_file_path_train, sound_file_path_validate, sound_file_path_test = np.split(sound_file_paths, [int(len(sound_file_paths)*0.8), int(len(sound_file_paths)*0.9)])
    transcription_train, transcription_validate, transcription_test = np.split(sound_file_transcriptions, [int(len(sound_file_transcriptions)*0.8), int(len(sound_file_transcriptions)*0.9)])
    
                    
    assert len(sound_file_path_train) == len(transcription_train),  "train split 길이 안맞음."
    assert len(sound_file_path_test) == len(transcription_test),  "test split 길이 안맞음."
    assert len(sound_file_path_validate) == len(transcription_validate),  "validate split 길이 안맞음."
                    
    with open(f"{os.path.join(args.asr_dest_folder, 'asr_split','train.tsv')}", "a+", encoding="utf-8") as asr_train, \
        open(f"{os.path.join(args.asr_dest_folder, 'asr_split','test.tsv')}", "a+", encoding="utf-8") as asr_test, \
        open(f"{os.path.join(args.asr_dest_folder, 'asr_split','validation.tsv')}", "a+", encoding="utf-8") as asr_validate:
        for i in range(len(sound_file_path_train)-1):
            asr_train.write(f"{sound_file_path_train[i]} :: {transcription_train[i]}\n")
        for i in range(len(sound_file_path_test)-1):
            asr_test.write(f"{sound_file_path_test[i]} :: {transcription_test[i]}\n")
        for i in range(len(sound_file_path_validate)-1):    
            asr_validate.write(f"{sound_file_path_validate[i]} :: {transcription_validate[i]}\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_dest_folder", type=str, required=True, help="folder that will contain all the data for asr model")
    parser.add_argument("--jsons", type=str, required=True, help="folder path that has json files inside of it")
    args = parser.parse_args()
    main(args)
    

