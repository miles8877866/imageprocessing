# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import cv2 

img = cv2.imread('edge.bmp')
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(img,kernel,iterations = 1)

cv2.imshow('new_edge.bmp', dilation)
cv2.imshow('img.bmp', img)
cv2.waitKey()
cv2.destroyAllWindows()
