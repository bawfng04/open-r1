
import os
import re

file_path = '/network-volume/envs/msb_grpo/lib/python3.10/site-packages/lighteval/logging/evaluation_tracker.py'

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
found = False
for line in lines:
    # Tìm dòng định nghĩa output_file_details
    if 'output_file_details = output_dir_details_sub_folder / f"details_{task_name}_{date_id}.parquet"' in line:
        indent = line[:line.find('output_file_details')]
        new_lines.append(f'{indent}sanitized_task_name = task_name.replace("|", "_")\n')
        new_lines.append(f'{indent}output_file_details = output_dir_details_sub_folder / f"details_{{sanitized_task_name}}_{{date_id}}.parquet"\n')
        found = True
    else:
        new_lines.append(line)

if found:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Successfully patched Lighteval evaluation_tracker.py")
else:
    print("Could not find the target line in evaluation_tracker.py. It might be already patched or using a different version.")
