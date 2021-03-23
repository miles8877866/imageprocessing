# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import cv2 

#讀檔
img = cv2.imread('lena512_8bit.bmp')

###low_pass_filter
kernal = np.ones([3, 3], np.float32)/9 
new_img = cv2.filter2D(img, -1, kernal);

cv2.imshow('lena512_8bit.bmp', img)
cv2.waitKey()
cv2.imshow('new_image.bmp', new_img)
cv2.waitKey()
cv2.destroyAllWindows()
