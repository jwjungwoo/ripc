# 지금까지의 label 파일에서 필요 없는 내용 제거 (annotations의 attributes, time)

import json
import os

label_dir = 'dataset/labels'
file_names = [os.path.splitext(file)[0] for file in os.listdir(label_dir)]

for i, file_name in enumerate(file_names):
    with open('dataset/labels/{}.json'.format(file_name), 'r', encoding='utf8') as f:
        current_json = json.load(f)

    new_json = dict()
    new_annotations = list()
    for annotation in current_json['annotations']:
        current_annotation = dict()
        current_annotation['label'] = annotation['attributes'][annotation['label']]
        current_annotation['points'] = annotation['points']
        new_annotations.append(current_annotation)

    new_json['annotations'] = new_annotations
    with open('dataset/labels/{}.json'.format(file_name), 'w', encoding='utf8') as f:
        json.dump(new_json, f)

    print(((i + 1) / len(file_names)) * 100)