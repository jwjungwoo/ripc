import numpy as np
import cv2

# 역할 : 특정 채널의 이미지를 받아서 임계값에 해당하는 픽셀만 추출한 이미지를 생성하여 반환함
# 인자 : ch - 특정 채널의 이미지, thresh - 채널 임계값
# 리턴 값 : 임계값 범위에 있는 값만 255로 설정한 이미지
def ch_thresh(ch, thresh=(80, 255)):
    binary = np.zeros_like(ch)
    binary[(ch > thresh[0]) & (ch <= thresh[1])] = 255  # 임계값 범위에 해당하는 픽셀의 값을 255로 설정

    return binary

# main에서 호출하는 함수
# 역할 : 차선 부분의 이미지를 받아 HSL 연산한 결과 이미지를 반환함
# 인자 : img - 현재 비디오 프레임의 이미지 중 차선이 있는 하위 40% 부분,
#        th_h - H 채널의 임계값, th_l - L 채널의 임계값, th_s - S 채널의 임계값
# 리턴 값 : HSL 연산을 한 결과 이미지
def hls_combine(img, th_h, th_l, th_s):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # 이미지를 BGR 채널에서 HSL 채널로 변경

    H = hls[:, :, 0]  # 이미지에서 각 픽셀의 H 값만 뽑아오기
    L = hls[:, :, 1]  # 이미지에서 각 픽셀의 L 값만 뽑아오기
    S = hls[:, :, 2]  # 이미지에서 각 픽셀의 S 값만 뽑아오기

    h_img = ch_thresh(H, th_h)  # H 채널에 대한 연산
    l_img = ch_thresh(L, th_l)  # L 채널에 대한 연산
    s_img = ch_thresh(S, th_s)  # S 채널에 대한 연산

    hls_combine = np.zeros_like(s_img).astype(np.uint8)
    # H 채널과 L 채널과 S 채널 결합
    # 1번째 부분 : S 채널이 특정 임계값을 넘고 (1 초과), L 채널이 0인 경우
    # 2번째 부분 : S 채널이 0이고, H 채널과 L 채널이 특정 임계값을 넘는 (1 초과) 경우
    hls_combine[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1))] = 255

    return hls_combine


def combine_hsl_canny(binary_img_hls, binary_img_canny):
    binary_img = (binary_img_hls + binary_img_canny)/2
    binary_img[binary_img != 0] = 255

    return binary_img