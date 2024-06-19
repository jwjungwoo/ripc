# 레이블 json 파일에서 필요 없는 내용들 제거
# 1920x1080에 맞춰져있는 annotation 정보를 1280x720에 맞게 수정

import json
import os

label_dir = 'dataset/labels'
label_file_names = [os.path.splitext(file)[0] for file in os.listdir(label_dir)]

for i, label_file in enumerate(label_file_names):
    with open('dataset/labels/{}.json'.format(label_file), 'r', encoding='utf8') as f:
        current_json = json.load(f)

    new_json = dict()
    annotations = list()
    for annotation in current_json['annotations']:
        current_annotation = dict()
        current_annotation['attributes'] = annotation['attributes']
        # 1920x1080 -> 1280x720
        new_points = list()
        for point in annotation['points']:
            x = point[0]
            y = point[1]
            new_x = int(x*(2/3))
            new_y = int(y*(2/3))
            new_points.append([new_x, new_y])
        current_annotation['points'] = new_points
        current_annotation['label'] = annotation['label']
        annotations.append(current_annotation)

    new_json['annotations'] = annotations
    new_json['time'] = current_json['info']['time']

    with open('dataset/labels/{}.json'.format(label_file), 'w', encoding='utf8') as f:
        json.dump(new_json, f)

    print(((i+1) / len(label_file_names)) * 100)