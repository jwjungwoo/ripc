# label은 있는데 attributes에는 아무것도 없는게 있어서 아무 것도 없는 annotation 제거

import json
import os

label_dir = 'dataset/labels'
label_file_names = [os.path.splitext(file)[0] for file in os.listdir(label_dir)]

for i, label_file in enumerate(label_file_names):
    with open('dataset/labels/{}.json'.format(label_file), 'r', encoding='utf8') as f:
        current_json = json.load(f)

    annotations = current_json['annotations']
    time = current_json['time']
    new_annotations = list()
    for annotation in annotations:
        if len(annotation['attributes']) != 0:
            new_annotations.append(annotation)

    new_json = dict()
    new_json['annotations'] = new_annotations
    new_json['time'] = time

    with open('dataset/labels/{}.json'.format(label_file), 'w', encoding='utf8') as f:
        json.dump(new_json, f)

    print(((i+1) / len(label_file_names)) * 100)