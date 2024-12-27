import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 (peppers.bmp)
img = cv2.imread('peppers.bmp')

# 2. YCbCr로 변환
img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# 3. Y 성분에 대해 히스토그램 균등화
y, cb, cr = cv2.split(img_ycbcr)
y_eq = cv2.equalizeHist(y)

# 4. 균등화된 Y 성분을 다시 합쳐서 BGR로 변환
img_y_eq = cv2.merge([y_eq, cb, cr])
img_y_eq_bgr = cv2.cvtColor(img_y_eq, cv2.COLOR_YCrCb2BGR)

# 5. RGB 각 채널에 대해 히스토그램 균등화
r, g, b = cv2.split(img)
r_eq = cv2.equalizeHist(r)
g_eq = cv2.equalizeHist(g)
b_eq = cv2.equalizeHist(b)

# 6. 균등화된 RGB 채널을 다시 합치기
img_rgb_eq = cv2.merge([b_eq, g_eq, r_eq])

# 7. 결과 비교: 원본 이미지, YCbCr에서 Y 성분 균등화된 이미지, RGB 채널별 균등화 이미지
titles = ['Original Image', 'Equalized Y(YCbCr)', 'Equalized RGB']
images = [img, img_y_eq_bgr, img_rgb_eq]

# 8. 결과 출력(1) - 한번에 보기
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.show()
# 8. 결과 출력(2) - 따로 보기
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.imshow('Original Image', img_y_eq_bgr)
cv2.waitKey(0)
cv2.imshow('Original Image', img_rgb_eq)
cv2.waitKey(0)

