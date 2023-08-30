import os 
import re 
pattern = re.compile("\([^\/]+\)\/\([^\/\(\)]+\)")
bracket_pattern = re.compile("[\(\)]")
for root, _dir, files in os.walk("./"):
    for file in files:
        fname, ext = os.path.splitext(file)
        if ext == ".tsv":
            with open(os.path.join(root, file), "r+", encoding="utf-8") as original_file, open(os.path.join(root, f"{fname}_refined.tsv"), "w+", encoding="utf-8") as refined_file:
                lines = original_file.readlines()
                new_lines = []
                for line in lines: 
                    _path, target_text = line.split(" :: ")
                    matches = re.findall(pattern, target_text)
                    if matches:
                        for item in matches:
                            first_part = item.split("/")[0]
                            first_part = re.sub(bracket_pattern, "", first_part)
                            line = line.replace(item, first_part)
                        new_lines.append(line)
                    else:
                        new_lines.append(line)
                for new in new_lines:
                    refined_file.write(f"{new}")
                    
                
                
                        
