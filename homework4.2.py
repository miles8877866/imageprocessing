# -*- coding: utf-8 -*-
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob as gb
import math

pathi = gb.glob("C:\\Users\\as722\\Desktop\\tests\\day0713\\data\\hw5_dataset\\j_training\\*.bmp")
img = io.imread('C:\\Users\\as722\\Desktop\\tests\\day0713\\data\\hw5_dataset\\i_testing\\obj14__21.bmp')
h, w = img.shape
data_bin = 9
histogram = np.zeros(576, dtype='float')#8*8*data_bin
hist = np.zeros([data_bin], dtype='float')
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

def bin_class(angle, magnitude):
    angle2 = angle // 20
    if angle2 > 8:
       angle2 = angle2 - 9
       hist[int(angle2)]+=magnitude
      
def dist(inputa, inputb):
    # value=0.
    value = (inputa - inputb)**2
    return value

index=0
for row in range(0, h, 16):
    for col in range(0, w, 16):
        hist = np.zeros([data_bin], dtype='float')
        for i in range(0, 16):
            for j in range(0, 16):
                #print(mag[row+i][col+j])
                bin_class(angle[row+i][col+j], mag[row+i][col+j])
            #normalize
            sum = np.sum(hist)
            #print(sum)
            if sum==0:
                hist = hist
            else:
                hist = hist/sum 
                histogram[index:index+9] = hist
        index=index+9
        

data = np.zeros((420, 576), dtype='float')
result = np.zeros([420], dtype='float')

for k in range(len(pathi)):
    img_train = io.imread(pathi[k])
    h, w = img_train.shape
    data_bin = 9
    histogramT = np.zeros(576, dtype='float')#8*8*data_bin
    hist = np.zeros([data_bin], dtype='float')
    gx = cv2.Sobel(img_train, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img_train, cv2.CV_32F, 0, 1, ksize=1)
    mag2, angle2 = cv2.cartToPolar(gx, gy, angleInDegrees=True)
      
    index=0
    for row in range(0, h, 16):
        for col in range(0, w, 16):
            hist = np.zeros([data_bin], dtype='float')
            for i in range(0, 16):
                for j in range(0, 16):
                    #print(mag[row+i][col+j])
                    bin_class(angle2[row+i][col+j], mag2[row+i][col+j])
                    #normalize
                sum = np.sum(hist)
                #print(sum)
                if sum==0:
                    hist = hist
                else:
                    hist = hist/sum 
                    histogramT[index:index+9] = hist
            index=index+9
    data[k][:] = histogramT
    
for i in range(len(pathi)):
    for j in range(576):
        result[i] += dist(data[i][j], histogram[j])
    result[i] = math.sqrt(result[i])
    
print(result)
result_x = np.zeros([420])       
result_x = np.argsort(result)

print(result_x[0:5])

for i in range(9):
    x = result_x[i]
    plt.figure()
    io.imshow(pathi[x])
    
io.show()
# plt.bar(np.arange(576), histogram, width = 1, align = 'edge')
# plt.show()


    
    
    