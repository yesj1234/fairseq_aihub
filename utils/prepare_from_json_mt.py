#json annotations: 
#{
#    "contents": {
#        "source": "라이브 스트리밍 출처",
#        "category": "라이브 스트리밍 카테고리",
#        "solved_copyright": "저작권",
#        "origin_lang" : "라이브 스트리밍 콘텐츠 언어"
#    },
#    "file": {
#        "source_filename":"라이브 스트리밍 콘텐츠 원본 데이터 파일명",
#        "source_filepath":"라이브 스트리밍 콘텐츠 원본 데이터 파일주소",
#        "sound_filename":"라이브 스트리밍 콘텐츠의 음성 추출 데이터 파일명: list",
#        "sound_filepath":"라이브 스트리밍 콘텐츠의 음성 추출 데이터 파일주소: list",
#        "start_sound_time":"음성 추출 데이터 파일의 시작 시간",
#        "end_sound_time": "음성 추출 데이터 파일의 종료 시간",
#        "start_voice_time":"목소리 추출 데이터의 시작 시간",
#        "end_voice_time":"라이브 스트리밍 플랫폼 정보"
#    },
#    "live streaming": {
#        "platform_info":"라이브 스트리밍 플랫폼 정보",
#        "subject":"라이브 스트리밍 주제",
#        "summary":"라이브 스트리밍 요약 설명",
#        "speaker_info":"라이브 스트리밍 화자 정보 ",
#        "location":"라이브 스트리밍 장소 정보"
#    },
#    "transcription": {
#        "text": "전사 텍스트: list"
#    },
#    "translation": {
#        "trans_lang": "번역 언어",
#        "trans_text": "번역 텍스트: list ",
#        "back_trans_lang": "역번역 언어",
#        "back_trans_text": "역번역 텍스트: list"        
#    },
#    "special language expression": {
#        "new_word": "신조어",
#        "abbreviation_word": "축약어",
#        "slang": "비속어",
#        "mistake": "말실수",
#        "again": "재발화",
#        "interjection": "간투사"
#    },
#    "environment": {
#        "outside": "실외",
#        "inside": "실내"
#    }
#}
import json 
import os 
import argparse



def main(args):
    for root, dir, files in os.path.walk(args.jsons):
        if files:
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == ".json":
                    with open(file, "r") as json_file, open(args.dest_file, "a") as destination:
                        json_data = json.load(json_file)
                        transcriptions = json_data["transcription"]["trans_text"]
                        translations = json_data["translation"]["trans_text"]
                        for i in range(len(transcriptions)):
                            destination.write(f"{transcriptions[i]} :: {translations[i]}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest-file", type=str, required=True, help="tsv file format that will eventually contain all the data")
    parser.add_argument("--jsons", type=str, required=True, help="folder path that has json files inside of it")
    args = parser.parse_args()
    main(args)