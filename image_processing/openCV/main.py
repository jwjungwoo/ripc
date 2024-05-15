import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

from finding_lines import *
from threshold import *

#### 직접 조정해야 하는 값  ############################################
th_h, th_l, th_s = (0, 45), (56, 255), (0, 70)  # HLS 필터의 threshold 값
canny_low, canny_high = 300, 753  # Canny 필터의 threshold 값
cut_ratio = 0.4  # 도로 부분만 자를 비율

# 원근 변환 (Perspective Transform) 좌표
src = np.float32([[588, 209], [304, 426], [689, 208], [989, 427]])
dst = np.float32([[200, 0], [200, 720], [520, 0], [520, 720]])

# 원근 변환 이미지의 1픽셀 당 m 거리 값
xm_per_pix = 1.5/320  # 320 픽셀이 1.5m
ym_per_pix = 1/720  # 720 픽셀이 1m

# 실제 m값 좌표를 기반으로 할 지 여부
is_real = True

##################################################################


# 왼쪽과 오른쪽 차선 객체 생성
left_line = Line()
right_line = Line()

video_width = 1280
video_height = 720

current_state = 'Not Lane'  # 'Lane'과 'Not Lane'
is_publish = False
to_lane_time = None  # 차선X -> 차선O 로 넘어가는 시간 체크
to_not_lane_time = None  # 차선O -> 차선X 로 넘어가는 시간 체크

cap = cv2.VideoCapture('source/test_video1.mp4')
#cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():  # 비디오의 한 프레임마다
    start_time = time.time()
    _, img = cap.read()
    img = cv2.resize(img, (1280,720))
    height, width = img.shape[:2]  # 이미지의 높이와 너비
    lane_img = img[int(cut_ratio * height):, :]  # 이미지의 하위 40%만 가져옴 (차선이 있는 부분)
    lane_height, lane_width = lane_img.shape[:2]

    # HSL 연산 수행
    binary_img_hls = hls_combine(lane_img, th_h, th_l, th_s)
    #binary_img_canny = cv2.Canny(lane_img, canny_low, canny_high)
    #binary_img = combine_hsl_canny(binary_img_hls, binary_img_canny)

    # 원근 변환한 이미지와 원근 변환 행렬, 역원근 변환 행렬 계산
    warp_img, M, Minv = warp_image(binary_img_hls, src, dst, (720, 720))
    # 차선 계산
    searching_img, lane_exist = find_LR_lines(warp_img, left_line, right_line, xm_per_pix, ym_per_pix, is_real)

    '''if lane_exist[0] or lane_exist[1]:  # 차선이 인식된 경우
        if current_state == 'Not Lane':
            if to_lane_time is None:
                to_lane_time = time.time()
                is_publish = False
            else:
                time_passed = time.time() - to_lane_time
                if time_passed < 1:
                    is_publish = False
                else:
                    is_publish = True
                    to_lane_time = None
                    current_state = 'Lane'
                    # 차선 인식됨 넘겨주기
                    print('\n\n차선 시작!!!!!\n\n')
        elif current_state == 'Lane':
            is_publish = True
    else:  # 차선이 인식되지 않은 경우
        if current_state == 'Not Lane':
            is_publish = False
        elif current_state == 'Lane':  # 직선 조향
            path = list()
            path.append(np.linspace(0, 1, 20))
            path.append([0.0] * 20)
            if to_not_lane_time is None:
                to_not_lane_time = time.time()
                is_publish = True
            else:
                time_passed = time.time() - to_not_lane_time
                if time_passed < 1:
                    is_publish = True
                else:
                    to_not_lane_time = None
                    current_state = 'Not Lane'
                    is_publish = False
                    # 차선 끊김 넘겨주기
                    print('\n\n차선 끝!!!!!\n\n')'''

    # 원근 변환한 이미지에 차선 부분을 표현한 이미지 계산
    w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line, lane_exist)

    # 검정색 바탕에 차선 부분만 표현한 이미지를 역원근 변환
    color_result = cv2.warpPerspective(w_color_result, Minv, (width, lane_height))
    lane_color = np.zeros_like(img)  # 전체 이미지와 동일한 형태의 0으로 채워진 넘파이 배열 (검정색 이미지)
    # 전체 이미지와 동일한 형태의 검정색 이미지에 차선 부분만 표현
    lane_color[int(cut_ratio * height):int(cut_ratio * height) + color_result.shape[0], :] = color_result

    # 현재 이미지에 차선 부분 표현을 더함
    result = cv2.addWeighted(img, 1, lane_color, 0.6, 0)

    #print('각도 :', angle)

    #print('걸린 시간 :', time.time() - start_time)
    #print()

    #cv2.imshow('result', result)
    #cv2.imshow('binary_img_hls', binary_img_hls)
    cv2.imshow('binary_img_canny', binary_img_hls)
    #cv2.imshow('binary_img', binary_img)
    cv2.imshow('warp_img', w_comb_result)
    cv2.imshow('result', result)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


"""
직접 조정해야 하는 값

main.py - hsl 필터 threshold 값
main.py - 원근 변환 좌표
finding_lines.py - 원근 변환 이미지의 픽셀 당 실제 m 값 (x좌표, y좌표)


"""


"""

# 원근 변환 좌표
src = np.float32([[550, 80], [220, 287], [720, 80], [960, 287]])  # 차선 부분 이미지의 source 좌표
dst = np.float32([[100, 0], [100, 700], [600, 0], [600, 700]])  # 원근 변환한 이미지의 destination 좌표

src = np.float32([[549, 35], [234, 221], [795, 35], [1062, 221]])  # 차선 부분 이미지의 source 좌표
dst = np.float32([[150, 0], [150, 700], [550, 0], [550, 700]])  # 원근 변환한 이미지의 destination 좌표


"""
