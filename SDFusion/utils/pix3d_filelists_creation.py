import os
import json
import random
from collections import defaultdict

def read_json_mapping(json_file_path: str):
    """Reads the JSON file and returns a dictionary mapping categories to image-model pairs."""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    category_mapping = defaultdict(list)
    for item in data:
        category = item['category']
        img_path = item['img']
        model_path = item['model']
        category_mapping[category].append((img_path, model_path))
    return category_mapping

def split_and_write_data(category_mapping: dict, output_dir: str, train_ratio=0.8):
    """Splits data for each category into training and testing sets and writes to files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category, pairs in category_mapping.items():
        random.shuffle(pairs)

        split_index = int(len(pairs) * train_ratio)
        train_pairs = pairs[:split_index]
        test_pairs = pairs[split_index:]

        write_pairs_to_file(train_pairs, os.path.join(output_dir, f'{category}_train.lst'))
        write_pairs_to_file(test_pairs, os.path.join(output_dir, f'{category}_test.lst'))

def write_pairs_to_file(pairs, file_name):
    """Writes pairs of filenames to a file."""
    with open(file_name, 'w') as file:
        for img, model in pairs:
            file.write(f"{model}\n")  # Writing only model path as per the requirement

# Usage
json_file_path = 'data/pix3d/pix3d.json'  # Update this with the actual path to the JSON file
output_directory = 'dataset_info_files/pix3d_filelists'  # Update this with the desired output path
category_mapping = read_json_mapping(json_file_path)
split_and_write_data(category_mapping, output_directory)
