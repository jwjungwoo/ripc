# 이미지 파일의 크기를 1920x1080에서 1280x720으로 줄임

import cv2
import os

image_dir = 'dataset/images'
image_file_names = [os.path.splitext(file)[0] for file in os.listdir(image_dir)]

for i, file in enumerate(image_file_names):
    img = cv2.imread('dataset/images/{}.png'.format(file))
    if img.shape == (720, 1280, 3):
        print(((i+1)/len(image_file_names))*100)
        continue

    print('변경 -', ((i+1) / len(image_file_names)) * 100)
    resized_img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('dataset/images/{}.png'.format(file), resized_img)

