import cv2
import sys
import numpy as np

# 1. BMP 이미지 파일 읽기
image = cv2.imread('BALLOON.bmp')

# 2. 이미지가 정상적으로 로드됐는지 확인
if image is None:
    sys.exit(' No file')

# 3. 이미지를 HSV 색 공간으로 변환
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 4. 파란색 범위를 정의 (HSV에서 파란색의 범위)
lower_blue = np.array([100, 150, 0])   # HSV에서 파란색의 하한값
upper_blue = np.array([140, 255, 255]) # HSV에서 파란색의 상한값

# 5. 파란색만 검출하는 마스크 생성
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# 6. 흰색 배경 이미지 생성 (같은 크기, 모든 픽셀 흰색)
white_background = np.ones_like(image, dtype=np.uint8) * 255

# 7. 파란색이 있는 부분만 원본 이미지에서 흰색 배경으로 복사
# 이때, cv2.copyTo 함수를 사용하여 흰색 배경에 파란색만 복사
result = cv2.copyTo(image, blue_mask, white_background)

# 8. 결과 이미지 저장
cv2.imwrite('blue_only_on_white_background_copyTo.bmp', result)

# 9. 결과 이미지 출력
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.imshow('Blue Only on White Background (copyTo)', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
