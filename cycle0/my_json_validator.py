import json
import os
from jsonschema import validate, ValidationError

import argparse
my_json_schema = {
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
                            "en_outside": {"type": "string"},
                            "en_inside": {"type": "string"},
                            "en_day": {"type": "string"},
                            "en_night": {"type": "string"}
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


def main(args):
    error_count = 0
    for root, dir, files in os.walk(args.json_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == ".json":
                try:
                    print(f"validating {file}")
                    validate(os.path.join(root, file), my_json_schema)
                except ValidationError as e:
                    print(f"error : {e} from {file}")
                    error_count += 1
                    pass
    print(error_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", help="검사할 json 파일들 들어 있는 루트 폴더 경로")
    args = parser.parse_args()

    main(args)
