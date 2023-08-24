import json
import os
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator
import argparse
# 1. required property 정하기.
# 2. 밸류 값으로 들어오는게 정해져 있는 경우 구분하기 => enum
my_json_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "contents": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "category": {"type": "string"},  # enum으로 넣을 수 있음?
                "solved_copyright": {"type": "string"},
                "origin_lang": {"type": "string"},  # enum으로 넣을 수 있음?
                "fi_source_filename": {"type": "string"},
                "fi_source_filepath": {"type": "string"},
                "li_platform_info": {"type": "string"},
                "li_subject": {"type": "string"},  # not required
                "li_summary": {"type": "string"},  # not required
                "li_location": {"type": "string"},
                "text_info": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            # string으로 들어올 때 minLength, maxLength 정해주면 됟듯?
                            "fi_sound_filename": {"type": "string"},
                            # string으로 들어올 때 minLength, maxLength 정해주면 됟듯?
                            "fi_sound_filepath": {"type": "string"},
                            # string으로 들어올 때 minLength, maxLength 정해주면 됟듯?
                            "fi_start_sound_time": {"type": "string"},
                            # string으로 들어올 때 minLength, maxLength 정해주면 됟듯?
                            "fi_end_sound_time": {"type": "string"},
                            # string으로 들어올 때 minLength, maxLength 정해주면 됟듯?
                            "fi_start_voice_time": {"type": "string"},
                            # string으로 들어올 때 minLength, maxLength 정해주면 됟듯?
                            "fi_end_voice_time": {"type": "string"},
                            "li_speaker_info": {
                                "type": "object",
                                "properties": {
                                    # enum으로 넣을 수 있음?
                                    "gender": {"type": "string"},
                                    # enum으로 넣을 수 있음?
                                    "ageGroup": {"type": "string"}
                                },
                                "required": ["gender", "ageGroup"]
                            },
                            "tc_text": {"type": "string"},
                            # enum으로 넣을 수 있음?
                            "tl_trans_lang": {"type": "string"},
                            # string으로 들어올 때 minLength, maxLength 정해주면 됟듯?
                            "tl_trans_text": {"type": "string"},
                            # enum으로 넣을 수 있음?
                            "tl_back_trans_lang": {"type": "string"},
                            "tl_back_trans_text": {"type": "string"},
                            "sl_new_word": {"type": "array"},
                            "sl_abbreviation_word": {"type": "array"},
                            "sl_slang": {"type": "array",
                                         "items": {
                                             "type": "string"
                                         }
                                         },
                            "sl_mistake": {"type": "array",
                                           "items": {
                                               "type": "string"
                                           }
                                           },
                            "sl_again": {"type": "array",
                                         "items": {
                                             "type": "string"
                                         }
                                         },
                            "sl_interjection": {"type": "array",
                                                "items": {
                                                    "type": "string"
                                                }
                                                },
                            # "en_outside": {"type": "string", "minLength": 1}, #enum으로 넣을 수 있음?
                            "en_outside": {"type": "string", "enum": ["X", "O"]},
                            # enum으로 넣을 수 있음?
                            "en_inside": {"type": "string", "minLength": 1},
                            # enum으로 넣을 수 있음?
                            "en_day": {"type": "string", "minLength": 1},
                            # enum으로 넣을 수 있음?
                            "en_night": {"type": "string", "minLength": 1},
                            # "en_noise": {"type": "string"}
                        },
                        "required": [
                            "fi_sound_filename",
                            "fi_sound_filepath",
                            "fi_start_sound_time",
                            "fi_end_sound_time",
                            "fi_start_voice_time",
                            "fi_end_voice_time",
                            "li_speaker_info",
                            "tc_text",
                            "tl_trans_lang",
                            "tl_trans_text",
                            "tl_back_trans_lang",
                            "tl_back_trans_text",
                            "sl_new_word",
                            "sl_abbreviation_word",
                            "sl_slang",
                            "sl_mistake",
                            "sl_again",
                            "sl_interjection",
                            "en_outside",
                            "en_inside",
                            "en_day",
                            "en_night",
                            # "en_noise"
                        ]
                    }
                }
            },
            "required": [
                "source",
                "category",
                "solved_copyright",
                "origin_lang",
                "fi_source_filename",
                "fi_source_filepath",
                "li_platform_info",
                "li_subject",  # 필수 아님.
                "li_summary",  # 필수 아님.
                "li_location",
                "text_info"
            ]
        }
    },
    "required": ["contents"]
}


def main(args):
    json_files = []
    required_property_missing_file = []
    required_property_value_missing_file = []
    validator = Draft7Validator(my_json_schema)
    for root, dir, files in os.walk(args.json_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == ".json":
                json_files.append(os.path.join(root, file))
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as json_file:
                        parsed_json = json.load(json_file)
                    for error in sorted(validator.iter_errors(parsed_json), key=str):
                        print(
                            f"Message: {error.message} \nFile: {file} \nError source : {'.'.join([str(item) for item in error.absolute_path])}\n")
                        if "required property" in error.message:
                            required_property_missing_file.append(file)
                        else:
                            required_property_value_missing_file.append(
                                error.message)
                        # print(error.validator)
                        # print(error.validator_value)
                        # print(error.relative_schema_path)
                        # print(error.absolute_schema_path)
                        # print(error.absolute_path)
                        # print(error.json_path)
                        # print(error.context)
                except ValidationError as e:
                    print(e)
                    continue
    required_property_missing_file = set(required_property_missing_file)
    print(f"이빨 빠진 파일 개수: {len(required_property_missing_file)}")
    print(f"형식에 안맞는 밸류 개수: {len(required_property_value_missing_file)}")
    print(f"검사한 총 파일 개수: {len(json_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", help="검사할 json 파일들 들어 있는 루트 폴더 경로")
    args = parser.parse_args()

    main(args)
