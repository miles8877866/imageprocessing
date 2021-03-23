# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import cv2 

img = cv2.imread('edge.bmp')
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 2)

cv2.imshow('new_edge.bmp', erosion)
cv2.waitKey()
cv2.destroyAllWindows()

