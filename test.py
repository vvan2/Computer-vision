import cv2 as cv
import sys

img=cv.imread('test.jpeg')
if img is None:
    sys.exit(' No file')
    
    
small=cv.resize(dsize=(0,0), fx=0.5, fy=0.5)
print(small.shape)    

cv.imwrite('soccer_gray_small.jpg', small)

cv.imshow('Original', img)
cv.imshow('Gray Resized', small)

cv.waitKey()