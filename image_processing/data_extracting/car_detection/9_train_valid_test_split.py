import random
import os

train_ratio = 0.85
valid_ratio = 0.1
test_ratio = 0.05

label_dir = 'dataset/labels'
file_names = [name.replace('.txt', '') for name in sorted(os.listdir(label_dir))]

# 랜덤하게 선택
numbers = list(range(len(file_names)))

train_selection = random.sample(numbers, k=int(len(file_names) * train_ratio))
remaining_numbers = [num for num in numbers if num not in train_selection]

valid_selection = random.sample(remaining_numbers, k=int(len(file_names) * valid_ratio))
test_selection = [num for num in remaining_numbers if num not in valid_selection]

# 훈련 데이터셋
for num in train_selection:
    current_file_name = file_names[num]

    # 이미지 이동
    old_path = 'dataset/images/{}.png'.format(current_file_name)
    new_path = 'dataset/train/images/{}.png'.format(current_file_name)
    os.rename(old_path, new_path)

    # 라벨 이동
    old_path = 'dataset/labels/{}.txt'.format(current_file_name)
    new_path = 'dataset/train/labels/{}.txt'.format(current_file_name)
    os.rename(old_path, new_path)

# 검증 데이터셋
for num in valid_selection:
    current_file_name = file_names[num]

    # 이미지 이동
    old_path = 'dataset/images/{}.png'.format(current_file_name)
    new_path = 'dataset/valid/images/{}.png'.format(current_file_name)
    os.rename(old_path, new_path)

    # 라벨 이동
    old_path = 'dataset/labels/{}.txt'.format(current_file_name)
    new_path = 'dataset/valid/labels/{}.txt'.format(current_file_name)
    os.rename(old_path, new_path)

# 테스트 데이터셋
for num in test_selection:
    current_file_name = file_names[num]

    # 이미지 이동
    old_path = 'dataset/images/{}.png'.format(current_file_name)
    new_path = 'dataset/test/images/{}.png'.format(current_file_name)
    os.rename(old_path, new_path)

    # 라벨 이동
    old_path = 'dataset/labels/{}.txt'.format(current_file_name)
    new_path = 'dataset/test/labels/{}.txt'.format(current_file_name)
    os.rename(old_path, new_path)