# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
#讀檔
img = cv2.imread('lena512_8bit.bmp')
#-1為圖像原深度
x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize = 3)
y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize = 3)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow('origin', img)
cv2.waitKey()

cv2.imshow("absX", absX)
cv2.waitKey()

cv2.imshow("absY", absY)
cv2.waitKey()

cv2.imshow("Result", dst)
cv2.waitKey()

cv2.destroyAllWindows()