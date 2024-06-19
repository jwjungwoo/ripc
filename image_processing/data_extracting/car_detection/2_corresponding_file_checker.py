# exist_images와 exist_labels 폴더를 새로 생성하여 이미지와 라벨 파일 모두 있는 파일을 해당 폴더로 옮김

import unicodedata
import shutil
import os

image_dir = 'dataset/images'
label_dir = 'dataset/labels'

image_files = [os.path.splitext(file)[0] for file in os.listdir(image_dir)]
label_files = [os.path.splitext(file)[0] for file in os.listdir(label_dir)]

# 문자열 정규화 (맥 한글 문자열 깨짐 이슈 해결 위함)
for image_name in image_files:
    old_name = 'dataset/images/{}.png'.format(image_name)
    new_name = 'dataset/images/{}.png'.format(unicodedata.normalize('NFC', image_name))
    os.rename(old_name, new_name)

for label_name in label_files:
    old_name = 'dataset/labels/{}.json'.format(label_name)
    new_name = 'dataset/labels/{}.json'.format(unicodedata.normalize('NFC', label_name))
    os.rename(old_name, new_name)

image_files = [os.path.splitext(file)[0] for file in os.listdir(image_dir)]
label_files = [os.path.splitext(file)[0] for file in os.listdir(label_dir)]

exist_files = [file for file in image_files if file in label_files]

for i, file in enumerate(exist_files):
    # 이미지 이동
    source_path = 'dataset/images/{}.png'.format(file)
    destination_path = 'dataset/exist_images/{}.png'.format(file)
    shutil.move(source_path, destination_path)

    # 라벨 이동
    source_path = 'dataset/labels/{}.json'.format(file)
    destination_path = 'dataset/exist_labels/{}.json'.format(file)
    shutil.move(source_path, destination_path)

    print(((i + 1) / len(exist_files)) * 100)

# 코드 수행 완료 후 exist_images와 exist_labels를 images와 labels로 변경하고, 기존 images와 labels는 삭제하기 (추후 필요 시 코드로 작성하기)