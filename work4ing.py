import cv2 as cv
import numpy as np

# 이미지를 불러오기
img = cv.imread("hand_sample2.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 가우시안 블러 적용
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# Otsu 방법으로 이진화
binary = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

#print(f"Threshold value: {t}")

# 이진화된 이미지 출력
cv.imshow("Color", img)
cv.imshow("Blurred", blurred)  # 블러 처리된 이미지 출력
cv.imshow("Binary", binary)
cv.waitKey(0)


# 모폴로지 커널 설정 (3x3 크기의 사각형 커널)
#kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

# Erosion으로 전경 추출
#fg = cv.erode(binary, kernel, iterations=4)
##cv.imshow("Foreground", fg)
#cv.waitKey(0)

# Dilation으로 배경 추출
#bg = cv.dilate(binary, kernel, iterations=2)
#t, bg = cv.threshold(bg, 1, 128, cv.THRESH_BINARY_INV)
#cv.imshow("Background", bg)
#cv.waitKey(0)

# 전경과 배경을 합쳐 markers 생성
#markers = fg + bg
#cv.imshow("Markers", markers)
#cv.waitKey(0)

# markers 배열을 int32 타입으로 변환
#markers = np.int32(markers)

# watershed 알고리즘 적용
#markers = cv.watershed(img, markers)

# 경계선 표시
#dst = np.ones(img.shape, np.uint8) * 255
#dst[markers == -1] = 0
#cv.imshow("Watershed", dst)
#cv.waitKey(0)

# 원본 이미지에 경계선 표시
#img[markers == -1] = [255, 0, 0]  # 경계선에 빨간색을 표시
#cv.imshow("Color with Boundaries", img)
#cv.waitKey(0)

# markers를 0-255 사이로 클리핑하여 8비트로 변환
#markers = np.uint8(np.clip(markers, 0, 255))
#cv.imshow("Segmentation", markers)
#cv.waitKey(0)

#cv.destroyAllWindows()
