# {
#     "contents":{
#         "source":"플레이타운",                    // 라이브 스트리밍 출처
#         "category":"일상/소통",           // 라이브 스트리밍 카테고리
#         "solved_copyright":"DIA TV",          // 저작권
#         "origin_lang":"KO",               // 라이브 스트리밍 콘텐츠 언어
#         "fi_source_filename":"다크소울 한국1등 여성 고인물분이 저희집에 왔습니다ㄷㄷ.mp4",        // 라이브 스트리밍 콘텐츠 원본 데이터 파일명
#         "fi_source_filepath":"https://objectstorage.ap-seoul-1.oraclecloud.com/n/cnb97trxvnun/b/clive-resource/o/contents/다크소울 한국1등 여성 고인물분이 저희집에 왔습니다ㄷㄷ.mp4",        // 라이브 스트리밍 원본 데이터 파일주소
#         "li_platform_info":"DIA TV",          // 라이브 스트리밍 플랫폼 정보
#         "li_subject":"일상/소통",                // 라이브 스트리밍 주제
#         "li_summary":"다크소울 한국1등 여성 고인물분이 저희집에 왔습니다ㄷㄷ 다크소울에 대한 모든걸 전파해주는 여성 고인물의 플레이를 감상하세요",                // 라이브 스트리밍 요약 설명
#         "li_location":"실내",               // 라이브 스트리밍 장소 정보
#         "text_info":[{
#             "fi_sound_filename":"002000_다크소울 한국1등 여성 고인물분이 저희집에 왔습니다ㄷㄷ/17.894999836808147_18.929998336997297_002000_다크소울 한국1등 여성 고인물분이 저희집에 왔습니다ㄷㄷ.wav",     // 라이브 스트리밍 콘텐츠의 음성 추출 데이터 파일명
#             "fi_sound_filepath":"https://objectstorage.ap-seoul-1.oraclecloud.com/n/cnb97trxvnun/b/clive-resource/o/contents/다크소울 한국1등 여성 고인물분이 저희집에 왔습니다ㄷㄷ/002000_다크소울 한국1등 여성 고인물분이 저희집에 왔습니다ㄷㄷ/17.894999836808147_18.929998336997297_002000_다크소울 한국1등 여성 고인물분이 저희집에 왔습니다ㄷㄷ.wav",     // 라이브 스트리밍 데이터 파일주소
#             "fi_start_sound_time":"177.27999999999997",   					// 음성 추출 데이터 파일의 시작 시간
#             "fi_end_sound_time":"182.07999999999998",     			// 음성 추출 데이터 파일의 종료 시간
#             "fi_start_voice_time":"177.27999999999997",   					// 목소리 추출 데이터의 시작 시간
#             "fi_end_voice_time":"182.07999999999998",     			// 목소리 추출 데이터의 종료 시간
#             "li_speaker_info":{         						// 라이브 스트리밍 화자 정보
#                 "gender":"남성",            						// 라이브 스트리밍 화자 성별
#                 "ageGroup":"30대-50대 미만"           				// 라이브 스트리밍 화자 연령대
#             },
#             "tc_text":"그것도 없어가지고",               // 전사 텍스트
#             "tl_trans_lang":"ZH",         // 번역 언어
#             "tl_trans_text":"那个也没有",         // 번역 텍스트
#             "tl_back_trans_lang":"KO",    // 역번역 언어
#             "tl_back_trans_text":"그것도 없고",    // 연번역 텍스트
#             "sl_new_word":[],           // 신조어 (여러 개가 있을 수 있기에 List<String>)
#             "sl_abbreviation_word":[],  // 축약어 (여러 개가 있을 수 있기에 List<String>)
#             "sl_slang":[],              // 비속어 (여러 개가 있을 수 있기에 List<String>)
#             "sl_mistake":[],            // 말실수 (여러 개가 있을 수 있기에 List<String>)
#             "sl_again":[],              // 재발화 (여러 개가 있을 수 있기에 List<String>)
#             "sl_interjection":[],       // 간투사 (여러 개가 있을 수 있기에 List<String>)
#             "en_outside":"X",            // 실외 (O,X로 구분)
#             "en_insdie":"O",             // 실내 (O,X로 구분)
#             "en_day":"X",                // 오전 (O,X로 구분)
#             "en_night":"O"               // 오후 (O,X로 구분)
#         }]
#     }
# }
import json 
import os 
import argparse



def main(args):
    for root, dir, files in os.path.walk(args.jsons):
        if files:
            print(f"json files from {os.path.join(root, dir)}")
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == ".json":
                    with open(file, "r") as json_file, open(args.asr_dest_file, "a") as asr_destination, open(args.mt_dest_file, "a") as mt_destination:
                        json_data = json.load(json_file)
                        sound_filepaths = json_data["file"]["sound_filepath"]
                        transcriptions = json_data["transcription"]["text"]
                        translations = json_data["translation"]["trans_text"]
                        for i in range(len(sound_filepaths)):
                            asr_destination.write(f"{sound_filepaths[i]} :: {transcriptions[i]}\n")
                            mt_destination.write(f"{transcriptions[i]} :: {translations[i]}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-dest-file", type=str, required=True, help="tsv file format that contains all the data for asr model")
    parser.add_argument("--mt-dest-file", type=str, required=True, help="tsv file format that contains all the data for mt model")
    parser.add_argument("--jsons", type=str, required=True, help="folder path that has json files inside of it")
    args = parser.parse_args()
    main(args)