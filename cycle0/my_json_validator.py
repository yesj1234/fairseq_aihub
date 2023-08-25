import json
import os
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator
import argparse
my_json_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "contents": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "minLength": 1},
                "category": {"type": "string", "minLength": 1},
                "solved_copyright": {"type": "string", "minLength": 1},
                "origin_lang": {"type": "string", "minLength": 1},
                "fi_source_filename": {"type": "string", "minLength": 1},
                "fi_source_filepath": {"type": "string", "minLength": 1},
                "li_platform_info": {"type": "string", "minLength": 1},
                "li_subject": {"type": "string"},  # not required
                "li_summary": {"type": "string"},  # not required
                "li_location": {"type": "string", "minLength": 1},
                "text_info": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fi_sound_filename": {"type": "string", "minLength": 1},
                            "fi_sound_filepath": {"type": "string", "minLength": 1},
                            "fi_start_sound_time": {"type": "string", "minLength": 1},
                            "fi_end_sound_time": {"type": "string", "minLength": 1},
                            "fi_start_voice_time": {"type": "string", "minLength": 1},
                            "fi_end_voice_time": {"type": "string", "minLength": 1},
                            "li_speaker_info": {
                                "type": "object",
                                "properties": {
                                    "gender": {"type": "string"},
                                    "ageGroup": {"type": "string"}
                                },
                                "required": ["gender", "ageGroup"]
                            },
                            "tc_text": {"type": "string", "minLength": 1},
                            "tl_trans_lang": {"type": "string", "minLength": 1},
                            "tl_trans_text": {"type": "string", "minLength": 1},
                            "tl_back_trans_lang": {"type": "string", "minLength": 1},
                            "tl_back_trans_text": {"type": "string", "minLength": 1},
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
                            "en_outside": {"type": "string", "minLength": 1},
                            "en_inside": {"type": "string", "minLength": 1},
                            "en_day": {"type": "string", "minLength": 1},
                            "en_night": {"type": "string", "minLength": 1},
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
                            # "sl_new_word", # not required
                            # "sl_abbreviation_word", # not required
                            # "sl_slang", # not required
                            # "sl_mistake", # not required
                            # "sl_again", # not required
                            # "sl_interjection", # not required
                            "en_outside",
                            "en_inside",
                            "en_day",
                            "en_night",
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
                "li_location",
                # "li_suject", # not required
                # "li_summary", # not required 
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
