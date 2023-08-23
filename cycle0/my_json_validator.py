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
                "source": {"type": "string"},
                "category": {"type": "string"},
                "solved_copyright": {"type": "string"},
                "origin_lang": {"type": "string"},
                "fi_source_filename": {"type": "string"},
                "fi_source_filepath": {"type": "string"},
                "li_platform_info": {"type": "string"},
                "li_subject": {"type": "string"},
                "li_summary": {"type": "string"},
                "li_location": {"type": "string"},
                "text_info": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fi_sound_filename": {"type": "string"},
                            "fi_sound_filepath": {"type": "string"},
                            "fi_start_sound_time": {"type": "string"},
                            "fi_end_sound_time": {"type": "string"},
                            "fi_start_voice_time": {"type": "string"},
                            "fi_end_voice_time": {"type": "string"},
                            "li_speaker_info": {
                                "type": "object",
                                "properties": {
                                    "gender": {"type": "string"},
                                    "ageGroup": {"type": "string"}
                                },
                                "required": ["gender", "ageGroup"]
                            },
                            "tc_text": {"type": "string"},
                            "tl_trans_lang": {"type": "string"},
                            "tl_trans_text": {"type": "string"},
                            "tl_back_trans_lang": {"type": "string"},
                            "tl_back_trans_text": {"type": "string"},
                            "sl_new_word": {"type": "array"},
                            "sl_abbreviation_word": {"type": "array"},
                            "sl_slang": {"type": "array"},
                            "sl_mistake": {"type": "array"},
                            "sl_again": {"type": "array"},
                            "sl_interjection": {"type": "array"},
                            "en_outside": {"type": "string", "minLength": 1},
                            "en_inside": {"type": "string", "minLength": 1},
                            "en_day": {"type": "string", "minLength": 1},
                            "en_night": {"type": "string", "minLength": 1}
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
                            "en_night"
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
                "li_subject",
                "li_summary",
                "li_location",
                "text_info"
            ]
        }
    },
    "required": ["contents"]
}


# def main(args):
#     error_count = 0
#     # validator = Draft7Validator(my_json_schema)

#     for root, dir, files in os.walk(args.json_dir):
#         for file in files:
#             _, ext = os.path.splitext(file)
#             if ext == ".json":
#                 try:
#                     print(f"validating {file}")
#                     with open(os.path.join(root, file), "r", encoding='utf-8') as json_file:
#                         parsed_json = json.load(json_file)
#                     validate(instance=parsed_json, schema=my_json_schema)
#                 except ValidationError as e:
#                     print(e.message)
#                     continue

#     print(error_count)

def main(args):
    error_count = 0
    validator = Draft7Validator(my_json_schema)
    for root, dir, files in os.walk(args.json_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == ".json":
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as json_file:
                        parsed_json = json.load(json_file)
                    for error in sorted(validator.iter_errors(parsed_json), key=str):
                        print(error.message)
                        # print(error.validator)
                        # print(error.validator_value)
                        # print(error.relative_schema_path)
                        # print(error.absolute_schema_path)
                        # print(error.absolute_path)
                        print(
                            f"Error at {'.'.join([str(item) for item in error.absolute_path])}")
                        # print(error.json_path)
                        # print(error.context)
                        # for err in error.absolute_path:
                        #     print(
                        #         f"Error at: {'.'.join(str(item) for item in err)}")

                        # print(error.message)
                except ValidationError as e:
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", help="검사할 json 파일들 들어 있는 루트 폴더 경로")
    args = parser.parse_args()

    main(args)
