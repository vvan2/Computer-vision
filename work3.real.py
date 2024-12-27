import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
food_image = cv2.imread('puppy.bmp', cv2.IMREAD_GRAYSCALE)

# (y,x)=(30,40)에서 에지 강도와 그레디언트 방향을 구하는 코드
y, x = 130,180

# Sobel 필터 적용
sobel_x = cv2.Sobel(food_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(food_image, cv2.CV_64F, 0, 1, ksize=3)

# dy, dx 값
dx = sobel_x[y, x]
dy = sobel_y[y, x]

# 에지 강도 (magnitude)와 그레디언트 방향 (angle)
edge_strength = np.sqrt(dx**2 + dy**2)
gradient_direction = np.arctan2(dy, dx) * (180 / np.pi)

print(f"dx: {dx}, dy: {dy}")
print(f"Edge Strength: {edge_strength}")
print(f"Gradient Direction: {gradient_direction}°")

# 2. Sobel 에지 강도 맵 구하기
sobel_edge_strength = np.sqrt(sobel_x**2 + sobel_y**2)

plt.imshow(sobel_edge_strength, cmap='gray')
plt.title('Sobel Edge Strength Map')
plt.show()

# 3. 가우시안 스므딩 후 Canny 에지 구하기
gaussian_blurred = cv2.GaussianBlur(food_image, (3, 3), sigmaX=1, sigmaY=1)
canny_edges = cv2.Canny(gaussian_blurred, 100, 200)

plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges after Gaussian Smoothing')
plt.show()

# 4. checkerboard 이미지에서 허프라인 구하기
checkerboard_image = cv2.imread('checkerboard.jpg', cv2.IMREAD_GRAYSCALE)
canny_checkerboard = cv2.Canny(checkerboard_image, 50, 150)

# 허프라인 변환
lines = cv2.HoughLines(canny_checkerboard, 1, np.pi / 180, 100)

# 허프라인 그리기
result_image = cv2.cvtColor(checkerboard_image, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.imshow(result_image)
plt.title('Hough Lines on Checkerboard')
plt.show()
