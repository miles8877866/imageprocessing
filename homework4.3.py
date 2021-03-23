# -*- coding: utf-8 -*-
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob as gb
import math

pathi = gb.glob("C:\\Users\\as722\\Desktop\\tests\\day0713\\data\\hw5_dataset\\j_training\\*.bmp")
img = io.imread('C:\\Users\\as722\\Desktop\\tests\\day0713\\data\\hw5_dataset\\i_testing\\obj2__21.bmp')

def bin_class(angle, magnitude):
    hist2 = np.zeros([9])
    angle2 = angle // 20
    if angle2 > 8:
       angle2 = angle2 - 9
       hist2[int(angle2)]+=magnitude
    return hist2
 
def hog(image):
    h, w = image.shape
    data_bin = 9
    histogram = np.zeros(576)#8*8*data_bin
    hist = np.zeros([data_bin])
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)        
      
    index=0
    for row in range(0, h, 16):
        for col in range(0, w, 16):
        
            for i in range(0, 16):
                for j in range(0, 16):
                #print(mag[row+i][col+j])
                    hist = bin_class(angle[row+i][col+j], mag[row+i][col+j])
                #normalize
                sum = np.sum(hist)
                if sum==0:
                    hist = hist
                else:
                    hist = hist/sum 
                    histogram[index:index+9] = hist
            index=index+9
    return histogram
    
def dist(i, j):
    for k in range(0, 576):
      pass

inp = np.zeros([576])
inp = hog(img)
print(inp)
data = np.zeros((420, 576))
result = np.zeros([420])
# 
# for i in range(len(pathi)):
#     img_train = io.imread(pathi[i])
#     data[i] = hog(img_train)
#     result[i] = dist(inp, data[i])
plt.bar(np.arange(576), inp, width = 1, align = 'edge')
plt.show()