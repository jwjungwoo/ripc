# 여러 곳에 흩어져있는 이미지와 라벨 파일을 하나의 폴더에 옮김

import shutil
import os

def get_all_files_in_directory(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

directory_path = 'dataset'
all_files = get_all_files_in_directory(directory_path)

for i, file in enumerate(all_files):
    extension = os.path.splitext(file)[1]
    file_name = os.path.basename(file)

    if extension == '.json':
        source_path = file
        destination_path = 'dataset/labels/{}'.format(file_name)
        shutil.move(source_path, destination_path)
    elif extension == '.png':
        source_path = file
        destination_path = 'dataset/images/{}'.format(file_name)
        shutil.move(source_path, destination_path)

    print(((i + 1) / len(all_files)) * 100)