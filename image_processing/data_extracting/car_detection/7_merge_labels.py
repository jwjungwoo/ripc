# 버스(소형,대형) + 통학버스(소형,대형) -> 버스
# 성인(노인포함) + 어린이 -> 사람
# 전동휠/전동킥보드/전동휠체어, 경찰차, 구급차, 소방차 제거

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
        label = annotation['label']
        if label == '버스(소형,대형)' or label == '통학버스(소형,대형)':
            label = '버스'
        elif label == '성인(노인포함)' or label == '어린이':
            label = '사람'
        elif label == '기타특장차(견인차, 쓰레기차, 크레인 등)':
            label = '기타특장차'
        elif label in ['전동휠/전동킥보드/전동휠체어', '경찰차', '구급차', '소방차']:
            continue

        current_annotation['label'] = label
        current_annotation['points'] = annotation['points']

        new_annotations.append(current_annotation)

    new_json['annotations'] = new_annotations
    with open('dataset/labels/{}.json'.format(file_name), 'w', encoding='utf8') as f:
        json.dump(new_json, f)

    print(((i + 1) / len(file_names)) * 100)