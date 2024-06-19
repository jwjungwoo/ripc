# 한글로 되어 있는 파일 이름을 모두 영어로 변경
# 파일 이름이 한글로 되어있는 경우 opencv에서 인식하지 못하므로

from googletrans import Translator
import os

def is_korean(word):
    korean_range = range(0xAC00, 0xD7A4 + 1)
    return all(ord(char) in korean_range for char in word)

translator = Translator()

trans_dict = dict()

image_dir = 'dataset/images'
label_dir = 'dataset/labels'

file_names = [os.path.splitext(file)[0] for file in os.listdir(image_dir)]

for i, file_name in enumerate(file_names):
    file_words = file_name.split('_')
    res_words = list()
    for word in file_words:
        _is_korean = is_korean(word)
        if _is_korean:
            if word not in list(trans_dict.keys()):
                translated_word = translator.translate(word, src='ko', dest='en').text
                trans_dict[word] = translated_word
            else:
                translated_word = trans_dict[word]
            res_words.append(translated_word)
        else:
            res_words.append(word)

    file_name_eng = '_'.join(res_words)
    file_name_eng = file_name_eng.replace(' ', '_').replace('_-', '')

    # 이미지
    old_name = 'dataset/images/{}.png'.format(file_name)
    new_name = 'dataset/images/{}.png'.format(file_name_eng)
    os.rename(old_name, new_name)

    # 라벨
    old_name = 'dataset/labels/{}.json'.format(file_name)
    new_name = 'dataset/labels/{}.json'.format(file_name_eng)
    os.rename(old_name, new_name)

    print(((i + 1) / len(file_names)) * 100)

