import json
import os
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator
import argparse
my_json_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Generated schema for Root",
  "type": "object",
  "properties": {
    "contentsIdx": {
      "type": "string"
    },
    "source": {
      "type": "string"
    },
    "category": {
      "type": "string"
    },
    "solved_copyright": {
      "type": "string"
    },
    "origin_lang_type": {
      "type": "string"
    },
    "origin_lang": {
      "type": "string"
    },
    "contentsName": {
      "type": "string"
    },
    "fi_source_filename": {
      "type": "string"
    },
    "fi_source_filepath": {
      "type": "string"
    },
    "li_platform_info": {
      "type": "string"
    },
    "li_subject": {
      "type": "string"
    },
    "li_location": {
      "type": "string"
    },
    "fi_sound_filename": {
      "type": "string"
    },
    "fi_sound_filepath": {
      "type": "string"
    },
    "li_total_video_time": {
      "type": "string"
    },
    "li_total_voice_time": {
      "type": "string"
    },
    "li_total_speaker_num": {
      "type": "string"
    },
    "fi_start_voice_time": {
      "type": "string"
    },
    "fi_end_voice_time": {
      "type": "string"
    },
    "fi_duration_time": {
      "type": "string"
    },
    "tc_text": {
      "type": "string"
    },
    "tl_trans_lang": {
      "type": "string"
    },
    "tl_trans_text": {
      "type": "string"
    },
    "tl_back_trans_lang": {
      "type": "string"
    },
    "tl_back_trans_text": {
      "type": "string"
    },
    "speaker_tone": {
      "type": "string"
    },
    "sl_new_word": {
      "type": "array",
      "items": {
          "type": "string"
      }
    },
    "sl_abbreviation_word": {
      "type": "array",
      "items": {
          "type": "string"
      }
    },
    "sl_slang": {
      "type": "array",
      "items": {
          "type": "string"
      }
    },
    "sl_mistake": {
      "type": "array",
      "items": {
          "type": "string"
      }
    },
    "sl_again": {
      "type": "array",
      "items": {
          "type": "string"
      }
    },
    "sl_interjection": {
      "type": "array",
      "items": {
          "type": "string"
      }
    },
    "place": {
      "type": "string"
    },
    "en_outside": {
      "type": "string"
    },
    "en_insdie": {
      "type": "string"
    },
    "day_night": {
      "type": "string"
    },
    "en_day": {
      "type": "string"
    },
    "en_night": {
      "type": "string"
    },
    "speaker_gender_type": {
      "type": "string"
    },
    "speaker_gender": {
      "type": "string"
    },
    "speaker_age_group_type": {
      "type": "string"
    },
    "speaker_age_group": {
      "type": "string"
    }
  },
  "required": [
    # "contentsIdx", 
    "source", # required
    "category", # required
    "solved_copyright", # required
    # "origin_lang_type",  
    "origin_lang", # required
    # "contentsName",  
    "fi_source_filename", # required
    "fi_source_filepath", # required
    "li_platform_info", # required  
    # "li_subject",
    # "li_location",
    "fi_sound_filename", # required
    "fi_sound_filepath", # required
    "li_total_video_time", # required
    "li_total_voice_time", # required
    "li_total_speaker_num", # required
    "fi_start_voice_time", # required
    "fi_end_voice_time", # required
    "fi_duration_time",
    "tc_text", # required
    "tl_trans_lang", # required
    "tl_trans_text", # required
    "tl_back_trans_lang", # required
    "tl_back_trans_text", # required
    "speaker_tone", # required
    # "sl_new_word",
    # "sl_abbreviation_word",
    # "sl_slang",
    # "sl_mistake",
    # "sl_again",
    # "sl_interjection",
    # "place",
    "en_outside", # required
    "en_insdie", # required
    "day_night", # required
    "en_day", # required
    "en_night", # required
    "speaker_gender_type", # required
    "speaker_gender", # required
    "speaker_age_group_type", # required 
    "speaker_age_group"# required
  ]
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
    print(f"필수 항목 불충족 파일 개수: {len(required_property_missing_file)}")
    print(f"형식 불충족 밸류 개수: {len(required_property_value_missing_file)}")
    print(f"검사한 총 파일 개수: {len(json_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", help="검사할 json 파일들 들어 있는 루트 폴더 경로")
    args = parser.parse_args()

    main(args)
