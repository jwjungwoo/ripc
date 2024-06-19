# json 파일로 되어있는 라벨링 파일을 YOLO 포맷에 맞게 변환

import json
import os

label_dir = 'dataset/labels'
file_names = [os.path.splitext(file)[0] for file in os.listdir(label_dir)]

index = {'세단': 0, 'SUV/승합차': 1, '버스': 2, '트럭': 3, '기타특장차': 4, '오토바이': 5, '사람': 6, '자전거': 7}
image_width = 1280
image_height = 720

cnt_dict = dict()
cnt = 0
for i, file_name in enumerate(file_names):
    with open('dataset/labels/{}.json'.format(file_name), 'r', encoding='utf8') as f:
        current_json = json.load(f)

    fw = open('dataset/yolo_labels/{}.txt'.format(file_name), 'w')

    for annotation in current_json['annotations']:
        idx = index[annotation['label']]
        points = annotation['points']
        x = (points[2][0] + points[0][0]) / 2
        x /= image_width
        y = (points[2][1] + points[0][1]) / 2
        y /= image_height
        w = points[2][0] - points[0][0]
        w /= image_width
        h = points[2][1] - points[0][1]
        h /= image_height

        result = '{} {} {} {} {}\n'.format(idx, x, y, w, h)
        #print(result)
        fw.write(result)

    fw.close()

    print(((i + 1) / len(file_names)) * 100)
