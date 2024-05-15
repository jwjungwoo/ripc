import numpy as np
import cv2
import time

# region Line
class Line:
    def __init__(self):
        # 창의 폭 설정 (+/- margin)
        self.window_margin = 56
        # 지난 n 번의 반복에서 적합된 선의 x 값
        self.prevx = []
        # 가장 최근의 적합에 대한 다항식 계수
        self.current_fit = [np.array([False])]
        # 시작 x 값
        self.startx = None
        # 끝 x 값
        self.endx = None
        # 감지된 선 픽셀의 x 값
        self.allx = None
        # 감지된 선 픽셀의 y 값
        self.ally = None
        # 도로 정보
        self.road_inf = None  # 도로의 정보 (커브 방향)
        self.curvature = None  # 도로의 곡률 반지름
        self.deviation = None  # 도로에서 차의 위치 (편향 값)
# endregion Line

# region 원근 변환

# 역할 : 원근 변환된 이미지를 생성해서 반환함
# 인자 : img - 차선 부분 이미지를 edge 검출 연산 한 이미지,
#       src - 원본 이미지의 source 좌표, dst - 원근 변환 결과 이미지의 destination 좌표,
#       size - 원근 변환 결과 이미지의 크기
# 리턴 값 : warp_img - 원근 변환된 이미지, M - 원근 변환 행렬, Minv - 역원근 변환 행렬
def warp_image(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst)  # 원근 변환 행렬 계산
    Minv = cv2.getPerspectiveTransform(dst, src)  # 역원근 변환 행렬 계산
    # 투영 변환을 적용하여 원근 변환된 이미지 생성
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    # 원근 변환 이미지에서 차량 부분 제거
    warp_img[580:720, 240:440] = 0

    return warp_img, M, Minv

# endregion 원근 변환

# region 차선 찾기

# 역할 : 원근 변환한 이미지를 받아 차선을 검출함
# 인자 : b_img - (edge 검출 연산한 이미지에 대해) 원근 변환한 이미지,
#        left_line - 왼쪽 차선 객체, right_line - 오른쪽 차선 객체
#       xm_per_pix : x 1픽셀 당 m 값, ym_per_pix : y 1픽셀 당 m 값
# 리턴 값 : 원근 변환한 이미지에서 차선을 표현한 이미지
def find_LR_lines(b_img, left_line, right_line, xm_per_pix, ym_per_pix, is_real):
    # 원근 변환한 이미지에 대해 세로방향 하위 절반 부분만 가지고 히스토그램을 계산함
    # 히스토그램은 각 열의 픽셀 합계 (세로방향 픽셀 합계)
    histogram = np.sum(b_img[int(b_img.shape[0] / 2):, :], axis=0)

    # 1차원 이미지를 3차원으로 쌓음
    output = np.dstack((b_img, b_img, b_img)) * 255

    # 왼쪽 차선과 오른쪽 차선의 시작 x 좌표값 구하기
    midpoint = np.int32(histogram.shape[0] / 2)  # 히스토그램의 중간 부분 (이미지의 가로 방향)
    start_leftX = np.argmax(histogram[:midpoint])  # 왼쪽 부분에서 픽셀 값이 가장 많은 열을 왼쪽 차선의 시작 x값으로 설정
    start_rightX = np.argmax(histogram[midpoint:]) + midpoint  # 오른쪽 부분에서 픽세 값이 가장 많은 열을 오른쪽 차선의 시작 x값으로 설정

    num_windows = 10  # 슬라이딩 윈도우의 개수
    window_height = np.int32(b_img.shape[0] / num_windows)  # 슬라이딩 윈도우의 높이

    # 흑백 이미지 (채널=1)에서 0이 아닌 픽셀의 좌표를 찾음
    nonzero = b_img.nonzero()  # 0이 아닌 좌표값들의 리스트
    nonzeroy = np.array(nonzero[0])  # 0이 아닌 좌표값들의 y 좌표 리스트
    nonzerox = np.array(nonzero[1])  # 0이 아닌 좌표값들의 x 좌표 리스트

    # 왼쪽 차선과 오른쪽 차선의 현재 x 좌표를 시작 x 좌표로 설정
    current_leftX = start_leftX
    current_rightX = start_rightX

    # 윈도우 중심의 x좌표를 옮기기 위한 최소한의 차선 픽셀 수
    min_num_pixel = 50

    # 각 윈도우에서 찾은 차선 픽셀의 인덱스를 저장할 리스트 선언 및 초기화
    win_left_lane = []
    win_right_lane = []

    # 각 윈도우에 있는 픽셀 개수 리스트
    left_lane_pixel_num_list = list()
    right_lane_pixel_num_list = list()

    window_margin = left_line.window_margin  # 윈도우의 마진 (window_margin*2가 윈도우의 세로 길이)
    # 슬라이딩 윈도우를 하나씩 순회하며
    for window in range(num_windows):
        # 현재 윈도우의 위치와 크기를 정의하는 변수 선언 및 계산
        # 윈도우의 수직 위치
        win_y_low = b_img.shape[0] - (window + 1) * window_height  # y축 하단 좌표
        win_y_high = b_img.shape[0] - window * window_height  # y축 상단 좌표
        # 윈도우의 수평 위치
        win_leftx_min = current_leftX - window_margin  # 왼쪽 차선 x축 왼쪽 좌표
        win_leftx_max = current_leftX + window_margin  # 왼쪽 차선 x축 오른쪽 좌표
        win_rightx_min = current_rightX - window_margin  # 오른쪽 차선 x축 왼쪽 좌표
        win_rightx_max = current_rightX + window_margin  # 오른쪽 차선 x축 오른쪽 좌표

        # 원근 변환한 차선 이미지(3차원)에 현재 윈도우의 경계를 직사각형으로 그림
        cv2.rectangle(output, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)  # 왼쪽 차선
        cv2.rectangle(output, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)  # 오른쪽 차선

        # 현재 윈도우 내에서 non-zero 픽셀의 인덱스를 찾아 left_window_inds와 right_window_inds에 저장
        # 픽셀의 인덱스는 nonzeroy와 nonzerox의 인덱스 값
        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
                nonzerox <= win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
                nonzerox <= win_rightx_max)).nonzero()[0]

        # 찾은 픽셀의 인덱스를 각각의 리스트에 저장
        win_left_lane.append(left_window_inds)  # 왼쪽 차선
        win_right_lane.append(right_window_inds)  # 오른쪽 차선

        # 현재 왼쪽과 오른쪽 윈도우에 포함된 픽셀 개수 저장
        left_lane_pixel_num_list.append(len(left_window_inds))
        right_lane_pixel_num_list.append(len(right_window_inds))

        # 만약 현재 윈도우 내에서 찾은 픽셀의 개수가 min_num_pixel보다 많다면, 해당 픽셀들의 x 좌표의 평균 위치를 계산하여 다음 윈도우의 중심으로 조정함
        if len(left_window_inds) > min_num_pixel:
            current_leftX = np.int32(np.mean(nonzerox[left_window_inds]))
        if len(right_window_inds) > min_num_pixel:
            current_rightX = np.int32(np.mean(nonzerox[right_window_inds]))
    # 슬라이딩 윈도우 완료

    # 왼쪽 차선과 오른쪽 차선에서 찾은 차선 픽셀 인덱스들을 하나의 리스트로 결합함
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)

    # 슬라이딩 윈도우 내에 아무런 픽셀이 없는 윈도우의 개수
    left_window_zero_count = sum(1 for element in left_lane_pixel_num_list if element == 0)
    right_window_zero_count = sum(1 for element in right_lane_pixel_num_list if element == 0)

    # 아무런 픽셀이 없는 윈도우가 7개 이하인것까지만 차선이 있다고 판단
    left_lane_exist = left_window_zero_count <= 7
    right_lane_exist = right_window_zero_count <= 7

    # 양쪽 차선이 모두 인식된 경우
    if left_lane_exist and right_lane_exist:
        # 차선 픽셀의 x와 y좌표를 추출함
        leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]  # 왼쪽 차선
        rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]  # 오른쪽 차선

        # 왼쪽 차선과 오른쪽 차선의 픽셀 값을 변경하여 차선 표현
        output[lefty, leftx] = [255, 0, 0]  # 왼쪽 차선을 빨간색으로 표현
        output[righty, rightx] = [0, 0, 255]  # 오른쪽 차선을 파란색으로 표현

        # 각 차선의 픽셀 좌표에 대해 2차 다항식을 적합시켜 계수를 구함 - 2차 회귀식 구하기
        try:
            left_fit = np.polyfit(lefty, leftx, 2)  # 왼쪽 차선
        except TypeError:
            left_fit = left_line.current_fit
        try:
            right_fit = np.polyfit(righty, rightx, 2)
        except TypeError:
            right_fit = right_line.current_fit

        # 현재 추정된 왼쪽 차선과 오른쪽 차선의 2차 다항식 계수를 각각 저장함
        left_line.current_fit = left_fit  # 왼쪽 차선
        right_line.current_fit = right_fit  # 오른쪽 차선

        # 플로팅 x값을 계산하기 위한 배열 생성 (0 ~ 세로 길이 - 1 까지의 배열 (1간격))
        ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

        # 각 차선에 대한 플로팅 x값 계산
        # 이미지의 모든 y좌표에 대해서 각 y좌표에 대한 x좌표의 값 (차선에 대한)
        left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]  # 왼쪽 차선
        right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]  # 오른쪽 차선

        # 차선 좌표값을 각 이전 차선 정보 리스트에 추가
        left_line.prevx.append(left_plotx)  # 왼쪽 차선
        right_line.prevx.append(right_plotx)  # 오른쪽 차선

        # 이전 차선 정보가 5개가 넘는 경우 - 이전 차선 정보 일부를 이용하여 평균낸 값을 사용함
        # 이전 차선 정보가 5개 이하인 경우 - 현재 인식된 차선 정보를 그대로 사용함
        if len(left_line.prevx) > 5:  # 왼쪽 차선에 저장된 이전 차선 정보의 개수가 5개가 넘는 경우
            left_avg_line = smoothing(left_line.prevx, 5)
            left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
            left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
            left_line.current_fit = left_avg_fit
            left_line.allx, left_line.ally = left_fit_plotx, ploty
        else:  # 왼쪽 차선에 저장된 이전 차선 정보의 개수가 10개 이하인 경우
            left_line.current_fit = left_fit
            left_line.allx, left_line.ally = left_plotx, ploty

        if len(right_line.prevx) > 5:  # 오른쪽 차선에 저장된 이전 차선 정보의 개수가 5개가 넘는 경우
            right_avg_line = smoothing(right_line.prevx, 5)
            right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
            right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
            right_line.current_fit = right_avg_fit
            right_line.allx, right_line.ally = right_fit_plotx, ploty
        else:  # 오른쪽 차선에 저장된 이전 차선 정보의 개수가 10개 이하인 경우
            right_line.current_fit = right_fit
            right_line.allx, right_line.ally = right_plotx, ploty

        # 왼쪽과 오른쪽 차선의 시작 x값을 원근 변환한 이미지의 맨 아래의 x값으로 설정
        left_line.startx, right_line.startx = left_line.allx[len(left_line.allx) - 1], right_line.allx[
            len(right_line.allx) - 1]
        # 왼쪽과 오른쪽 차선의 끝 x값을 원근 변환한 이미지의 맨 위의 x값으로 설정
        left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

        #path = get_path_list(left_line, right_line, xm_per_pix, ym_per_pix, 'both', is_real)
        calculate_angle(left_line, right_line, xm_per_pix, ym_per_pix, detect_status='both', is_real=False)

        # 곡률 반지름 계산
        #curverad = rad_of_curvature(left_line, right_line, xm_per_pix, ym_per_pix, 'both', is_real)

    # 왼쪽 차선만 인식된 경우
    elif left_lane_exist and not right_lane_exist:
        # 차선 픽셀의 x와 y좌표를 추출함
        leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]  # 왼쪽 차선

        # 왼쪽 차선과 오른쪽 차선의 픽셀 값을 변경하여 차선 표현
        output[lefty, leftx] = [255, 0, 0]  # 왼쪽 차선을 빨간색으로 표현

        # 각 차선의 픽셀 좌표에 대해 2차 다항식을 적합시켜 계수를 구함 - 2차 회귀식 구하기
        try:
            left_fit = np.polyfit(lefty, leftx, 2)  # 왼쪽 차선
        except TypeError:
            left_fit = left_line.current_fit

        # 현재 추정된 왼쪽 차선과 오른쪽 차선의 2차 다항식 계수를 각각 저장함
        left_line.current_fit = left_fit  # 왼쪽 차선
        right_line.current_fit = np.array([False])

        # 플로팅 x값을 계산하기 위한 배열 생성 (0 ~ 세로 길이 - 1 까지의 배열 (1간격))
        ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

        # 각 차선에 대한 플로팅 x값 계산
        # 이미지의 모든 y좌표에 대해서 각 y좌표에 대한 x좌표의 값 (차선에 대한)
        left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]  # 왼쪽 차선

        # 차선 좌표값을 각 이전 차선 정보 리스트에 추가
        left_line.prevx.append(left_plotx)  # 왼쪽 차선
        right_line.prevx.clear()

        # 이전 차선 정보가 10개가 넘는 경우 - 이전 차선 정보 일부를 이용하여 평균낸 값을 사용함
        # 이전 차선 정보가 10개 이하인 경우 - 현재 인식된 차선 정보를 그대로 사용함
        if len(left_line.prevx) > 5:  # 왼쪽 차선에 저장된 이전 차선 정보의 개수가 10개가 넘는 경우
            left_avg_line = smoothing(left_line.prevx, 5)
            left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
            left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
            left_line.current_fit = left_avg_fit
            left_line.allx, left_line.ally = left_fit_plotx, ploty
        else:  # 왼쪽 차선에 저장된 이전 차선 정보의 개수가 10개 이하인 경우
            left_line.current_fit = left_fit
            left_line.allx, left_line.ally = left_plotx, ploty

        # 왼쪽과 오른쪽 차선의 시작 x값을 원근 변환한 이미지의 맨 아래의 x값으로 설정
        left_line.startx = left_line.allx[len(left_line.allx) - 1]
        # 왼쪽과 오른쪽 차선의 끝 x값을 원근 변환한 이미지의 맨 위의 x값으로 설정
        left_line.endx = left_line.allx[0]

        #path = get_path_list(left_line, right_line, xm_per_pix, ym_per_pix, 'left_only', is_real)

        # 곡률 반지름 계산
        #curverad = rad_of_curvature(left_line, None, xm_per_pix, ym_per_pix, 'left_only', is_real)

    # 오른쪽 차선만 인식된 경우
    elif not left_lane_exist and right_lane_exist:
        # 차선 픽셀의 x와 y좌표를 추출함
        rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]  # 오른쪽 차선

        # 왼쪽 차선과 오른쪽 차선의 픽셀 값을 변경하여 차선 표현
        output[righty, rightx] = [0, 0, 255]  # 오른쪽 차선을 파란색으로 표현

        try:
            right_fit = np.polyfit(righty, rightx, 2)
        except TypeError:
            right_fit = right_line.current_fit

        # 현재 추정된 왼쪽 차선과 오른쪽 차선의 2차 다항식 계수를 각각 저장함
        right_line.current_fit = right_fit  # 오른쪽 차선
        left_line.current_fit = np.array([False])

        # 플로팅 x값을 계산하기 위한 배열 생성 (0 ~ 세로 길이 - 1 까지의 배열 (1간격))
        ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

        # 각 차선에 대한 플로팅 x값 계산
        # 이미지의 모든 y좌표에 대해서 각 y좌표에 대한 x좌표의 값 (차선에 대한)
        right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]  # 오른쪽 차선

        # 차선 좌표값을 각 이전 차선 정보 리스트에 추가
        right_line.prevx.append(right_plotx)  # 오른쪽 차선
        left_line.prevx.clear()

        if len(right_line.prevx) > 5:  # 오른쪽 차선에 저장된 이전 차선 정보의 개수가 10개가 넘는 경우
            right_avg_line = smoothing(right_line.prevx, 5)
            right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
            right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
            right_line.current_fit = right_avg_fit
            right_line.allx, right_line.ally = right_fit_plotx, ploty
        else:  # 오른쪽 차선에 저장된 이전 차선 정보의 개수가 10개 이하인 경우
            right_line.current_fit = right_fit
            right_line.allx, right_line.ally = right_plotx, ploty

        # 왼쪽과 오른쪽 차선의 시작 x값을 원근 변환한 이미지의 맨 아래의 x값으로 설정
        right_line.startx = right_line.allx[len(right_line.allx) - 1]
        # 왼쪽과 오른쪽 차선의 끝 x값을 원근 변환한 이미지의 맨 위의 x값으로 설정
        right_line.endx = right_line.allx[0]

        #path = get_path_list(left_line, right_line, xm_per_pix, ym_per_pix, 'right_only', is_real)
        # 곡률 반지름 계산
        #curverad = rad_of_curvature(None, right_line, xm_per_pix, ym_per_pix, 'right_only', is_real)

    else:  # 양 쪽 차선이 모두 인식되지 않은 경우
        left_line.prevx.clear()
        right_line.prevx.clear()

        left_line.current_fit = np.array([False])
        path = None
        #curverad = None

    return output, (left_lane_exist, right_lane_exist)  # 원근 변환한 이미지에 왼쪽 차선과 오른쪽 차선을 각각 다른 색깔로 표시한 이미지


# 역할 : 가장 최근 pre_lines 개의 차선 정보의 평균값을 계산함
# 인자 : lines - 특정 차선의 모든 이전 차선 정보 (각 y좌표에 대한 차선 x좌표의 값),
#        pre_lines - 사용할 이전 차선 정보의 개수
# 리턴 값 : 최근 차선의 y좌표에 대한 x좌표의 평균값 (1차원 넘파이 배열)
def smoothing(lines, pre_lines=3):
    # 리스트(리스트 내에 넘파이 배열)를 넘파이 배열(넘파이 배열 내에 넘파이 배열)로 바꿈
    lines = np.squeeze(lines)
    # 720개(원근 변환한 이미지의 높이)의 0을 가진 넘파이 배열 생성
    avg_line = np.zeros((720))

    # 각 차선 정보에 대해서
    # reversed(lines)를 사용해서 가장 최근의 차선 정보부터
    for ii, line in enumerate(reversed(lines)):
        if ii == pre_lines:  # pre_lines 만큼의 차선 정보를 이용했으면
            break  # 빠져나가기
        avg_line += line  # 차선 정보 더하기
    avg_line = avg_line / pre_lines  # 차선의 평균 값

    return avg_line

'''def get_path_list(left_line, right_line, xm_per_pix, ym_per_pix, detect_status, is_real):
    path_y = np.linspace(0, 720-1, 20)  # y축 값

    if is_real:  # 실제 m값 좌표 기반
        if detect_status == 'both':
            leftx, rightx = left_line.allx, right_line.allx  # x축 값

            # 왼쪽과 오른쪽 차선 x축 값을 역순으로 하여 저장
            leftx = leftx[::-1]  # 왼쪽 차선
            rightx = rightx[::-1]  # 오른쪽 차선

            center_x = (leftx + rightx) / 2
            center_fit = np.polyfit(path_y, center_x, 2)

            path_x = center_fit[0]*(path_y**2) + center_fit[1]*path_y + center_fit[2]
            path_x -= 360

            path_x *= xm_per_pix
            path_y *= ym_per_pix
            path_x[0] = 0.0
            path_y[0] = 0.0

        elif detect_status == 'left':
            leftx = left_line.allx  # x축 값

            # 왼쪽과 오른쪽 차선 x축 값을 역순으로 하여 저장
            leftx = leftx[::-1]  # 왼쪽 차선

            left_fit = np.polyfit(path_y, leftx, 2)
            path_x = left_fit[0]*(path_y**2) + left_fit[1]*path_y + left_fit[2]
            path_x -= 180

            path_x *= xm_per_pix
            path_y *= ym_per_pix
            path_x[0] = 0.0
            path_y[0] = 0.0

        elif detect_status == 'right':
            rightx = right_line.allx  # x축 값

            # 왼쪽과 오른쪽 차선 x축 값을 역순으로 하여 저장
            rightx = rightx[::-1]  # 오른쪽 차선
            right_fit = np.polyfit(path_y, rightx, 2)

            path_x = right_fit[0] * (path_y ** 2) + right_fit[1] * path_y + right_fit[2]
            path_x -= 540

            path_x *= xm_per_pix
            path_y *= ym_per_pix
            path_x[0] = 0.0
            path_y[0] = 0.0

        return [path_x, path_y]'''

# 역할 : 현재 인식된 차선에 대한 각도를 계산함
# 인자 : left_line - 왼쪽 차선 Line 객체, right_line - 오른쪽 차선 Line 객체
# 리턴 값 : 없음
def calculate_angle(left_line, right_line, xm_per_pix, ym_per_pix, detect_status='both', is_real=False):
    # 차선을 2차 곡선에 적합한 값
    ploty = np.linspace(0, 720-1, 720)  # y축 값
    angle_list = np.linspace(0, 720, 72 + 1)
    y = 10
    print('right_line :', right_line)

    if is_real:  # 실제 m값 좌표 기반
        angle_list *= ym_per_pix
        y *= ym_per_pix
        if detect_status == 'both':
            leftx, rightx = left_line.allx, right_line.allx  # x축 값

            # 왼쪽과 오른쪽 차선 x축 값을 역순으로 하여 저장
            leftx = leftx[::-1]  # 왼쪽 차선
            rightx = rightx[::-1]  # 오른쪽 차선

            center_x = (leftx + rightx) / 2
            center_fit = np.polyfit(ploty*ym_per_pix, center_x*xm_per_pix, 2)

            slope = 2 * center_fit[0] * y + center_fit[1]
            radian = np.arctan(slope)
            angle = radian * (180 / np.pi)
            slope_list = 2 * center_fit[0] * angle_list + center_fit[1]
            radian_list = np.arctan(slope_list)
            angle_list = radian_list * (180 / np.pi)
            # print('기울기 :', slope)
            # print('각도 : ', angle)

        elif detect_status == 'left_only':
            leftx = left_line.allx  # x축 값

            # 왼쪽과 오른쪽 차선 x축 값을 역순으로 하여 저장
            leftx = leftx[::-1]  # 왼쪽 차선
            left_fit = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)

            slope = 2 * left_fit[0] * y + left_fit[1]
            radian = np.arctan(slope)
            angle = radian * (180 / np.pi)
            slope_list = 2 * left_fit[0] * angle_list + left_fit[1]
            radian_list = np.arctan(slope_list)
            angle_list = radian_list * (180 / np.pi)

            # print('기울기 :', slope)
            # print('각도 : ', angle)

        elif detect_status == 'right_only':
            rightx = right_line.allx  # x축 값

            # 왼쪽과 오른쪽 차선 x축 값을 역순으로 하여 저장
            rightx = rightx[::-1]  # 오른쪽 차선
            right_fit = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

            slope = 2 * right_fit[0] * y + right_fit[1]
            radian = np.arctan(slope)
            angle = radian * (180 / np.pi)
            slope_list = 2 * right_fit[0] * angle_list + right_fit[1]
            radian_list = np.arctan(slope_list)
            angle_list = radian_list * (180 / np.pi)

    else:  # 이미지 좌표 기반
        if detect_status == 'both':
            leftx, rightx = left_line.allx, right_line.allx  # x축 값

            # 왼쪽과 오른쪽 차선 x축 값을 역순으로 하여 저장
            leftx = leftx[::-1]  # 왼쪽 차선
            rightx = rightx[::-1]  # 오른쪽 차선

            center_x = (leftx + rightx) / 2
            center_fit = np.polyfit(ploty, center_x, 2)

            slope = 2 * center_fit[0] * y + center_fit[1]
            radian = np.arctan(slope)
            angle = radian * (180 / np.pi)
            slope_list = 2 * center_fit[0] * angle_list + center_fit[1]
            radian_list = np.arctan(slope_list)
            angle_list = radian_list * (180 / np.pi)
            # print('기울기 :', slope)
            # print('각도 : ', angle)

        elif detect_status == 'left_only':
            leftx = left_line.allx  # x축 값

            # 왼쪽과 오른쪽 차선 x축 값을 역순으로 하여 저장
            leftx = leftx[::-1]  # 왼쪽 차선
            left_fit = np.polyfit(ploty, leftx, 2)

            slope = 2 * left_fit[0] * y + left_fit[1]
            radian = np.arctan(slope)
            angle = radian * (180 / np.pi)
            slope_list = 2 * left_fit[0] * angle_list + left_fit[1]
            radian_list = np.arctan(slope_list)
            angle_list = radian_list * (180 / np.pi)
            # print('기울기 :', slope)
            # print('각도 : ', angle)

        elif detect_status == 'right_only':
            rightx = right_line.allx  # x축 값

            # 왼쪽과 오른쪽 차선 x축 값을 역순으로 하여 저장
            rightx = rightx[::-1]  # 오른쪽 차선
            right_fit = np.polyfit(ploty, rightx, 2)

            slope = 2 * right_fit[0] * y + right_fit[1]
            radian = np.arctan(slope)
            angle = radian * (180 / np.pi)
            slope_list = 2 * right_fit[0] * angle_list + right_fit[1]
            radian_list = np.arctan(slope_list)
            angle_list = radian_list * (180 / np.pi)
        # print('기울기 :', slope)
        # print('각도 : ', angle)

    #print('각도 :', angle)

    angle_avg = np.mean(angle_list)

    return angle




    ''' 차선 좌표 출력
    for i in range(len(ploty)):
        print((leftx[i], ploty[i]))
    print()
    for i in range(len(ploty)):
        print((rightx[i], ploty[i]))
    time.sleep(123123)'''

# 역할 : 차선의 곡선 반지름 계산
# 인자 :
# 리턴 값 :
'''def rad_of_curvature(left_line, right_line, xm_per_pix, ym_per_pix, detect_status='both', is_real=False):
    ploty = np.linspace(0, 720-1, 720)
    
    if is_real:  # 실제 m 좌표 기반
        if detect_status == 'both':
            leftx, rightx = left_line.allx, right_line.allx

            leftx = leftx[::-1]
            rightx = rightx[::-1]

            left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
            right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

            y_eval = np.max(ploty)

            left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * left_fit_cr[0])
            right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * right_fit_cr[0])

            curverad = (left_curverad + right_curverad) / 2


        elif detect_status == 'left_only':
            leftx = left_line.allx
            leftx = leftx[::-1]
            left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)

            y_eval = np.max(ploty)

            curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * left_fit_cr[0])

        elif detect_status == 'right_only':
            rightx = right_line.allx
            rightx = rightx[::-1]
            right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

            y_eval = np.max(ploty)

            curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * right_fit_cr[0])

    else:  # 이미지 좌표 기반
        if detect_status == 'both':
            leftx, rightx = left_line.allx, right_line.allx
    
            leftx = leftx[::-1]
            rightx = rightx[::-1]
    
            left_fit_cr = np.polyfit(ploty, leftx, 2)
            right_fit_cr = np.polyfit(ploty, rightx, 2)
    
            y_eval = np.max(ploty)
    
            left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * left_fit_cr[0])
            right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * right_fit_cr[0])
    
            curverad = (left_curverad + right_curverad) / 2
    
    
        elif detect_status == 'left_only':
            leftx = left_line.allx
            leftx = leftx[::-1]
            left_fit_cr = np.polyfit(ploty, leftx, 2)
    
            y_eval = np.max(ploty)
    
            curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * left_fit_cr[0])
    
        elif detect_status == 'right_only':
            rightx = right_line.allx
            rightx = rightx[::-1]
            right_fit_cr = np.polyfit(ploty, rightx, 2)
    
            y_eval = np.max(ploty)
    
            curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * right_fit_cr[0])


    #print('곡면 반지름 :', curverad)
    return curverad'''





# endregion 차선 찾기

# region 차선 그리기

# main에서 호출하는 함수
# 역할 :
# 인자 : img - 원근 변환한 이미지에서 차선을 표현한 이미지,
#        left_line - 왼쪽 차선 Line 객체, right_line - 오른쪽 차선 Line 객체
#        lane_color - 차선의 색깔, road_color - 도로의 색깔
# 리턴 값 : result - 원근 변환한 이미지에 차선 부분을 표현한 이미지,
#           window_img - 검정색 이미지에 차선 부분을 표현한 이미지
def draw_lane(img, left_line, right_line, lane_exist, lane_color=(255, 0, 255), road_color=(0, 255, 0)):
    left_lane_exist = lane_exist[0]
    right_lane_exist = lane_exist[1]

    # img와 동일한 형태의, 0으로 채워진 넘파이 배열 (검정색으로 채워진 이미지)
    window_img = np.zeros_like(img)

    # 왼쪽 차선과 오른쪽 차선의 정보
    window_margin = left_line.window_margin  # 윈도우의 마진
    if left_lane_exist:
        left_plotx = left_line.allx
    if right_lane_exist:
        right_plotx = right_line.allx
    ploty = left_line.ally  # 차선의 y값

    # 왼쪽 차선의 왼쪽 부분과 오른쪽 부분의 x좌표
    if left_lane_exist:
        left_pts_l = np.array([np.transpose(np.vstack([left_plotx - window_margin / 5, ploty]))])  # 왼쪽
        left_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin / 5, ploty])))])  # 오른쪽
        left_pts = np.hstack((left_pts_l, left_pts_r))

    # 오른쪽 차선의 왼쪽 부분과 오른쪽 부분의 x좌표
    if right_lane_exist:
        right_pts_l = np.array([np.transpose(np.vstack([right_plotx - window_margin / 5, ploty]))])  # 왼쪽
        right_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plotx + window_margin / 5, ploty])))])  # 오른쪽
        right_pts = np.hstack((right_pts_l, right_pts_r))

    # 검정색 이미지에 차선 부분을 lane_color에 저장된 색깔로 채움
    if left_lane_exist:
        cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
    if right_lane_exist:
        cv2.fillPoly(window_img, np.int_([right_pts]), lane_color)

    # 차선 내부 도로 부분의 왼쪽 부분과 오른쪽 부분의 x좌표
    if left_lane_exist:
        pts_left = np.array([np.transpose(np.vstack([left_plotx + window_margin / 5, ploty]))])
    if right_lane_exist:
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx - window_margin / 5, ploty])))])

    if left_lane_exist and right_lane_exist:  # 양쪽 차선 모두 인식된 경우
        pts = np.hstack((pts_left, pts_right))

        # 검정색 이미지에 도로 부분을 road_color에 저장된 색깔로 채움
        cv2.fillPoly(window_img, np.int_([pts]), road_color)

    # 원근 변환한 이미지에 차선 부분을 표현한 이미지를 더해서 하나의 이미지로 결합함
    result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

    return result, window_img

# endregion 차선 그리기