# -*- coding: utf-8 -*-

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import math
import cv2
from skimage import feature as ft
import os, sys
import glob as gb



imgt = gb.glob('./data\\hw5_dataset\\j_training\\*.bmp')
# img = io.imread('./data\\hw5_dataset\\i_testing\\obj2__21.bmp')
img = io.imread('./data\\hw5_dataset\\j_training\\obj1__0.bmp')

def ab(angle):
    angleblock = angle//20
    angleblock = angleblock.astype(int)
    if angleblock>=9:
       angleblock = angleblock-9
       
    return angleblock
       
       
def binangle(lena):
    h,w = lena.shape
    data_bin = 9
    histogram = np.zeros(576)
    hist = np.zeros([data_bin])
    
    
    x = cv2.Sobel(lena, cv2.CV_32F, 1, 0, ksize=1)
    y = cv2.Sobel(lena, cv2.CV_32F, 0, 1, ksize=1)

    mag, angle = cv2.cartToPolar(x, y, angleInDegrees=True)
           
    
    
    count=0   
    for i in range (0, h, 16):
        for j in range (0, w, 16):
            hist = np.zeros([data_bin])
            for k in range (0, 16):
                for l in range (0, 16):
                    hist[ab(angle[k+i][l+j])]+= mag[k+i][l+j]
                    
      
            SUM = np.sum(hist)
           
            if SUM==0:
                hist=hist
            else:
                hist = hist/SUM
                
             
            for m in range(9):
                    histogram[count] = hist[m]
                    count+=1
                    
    return histogram                     
            
            
input1=binangle(img)

k=0
comp = np.zeros(420,dtype=float)

for file in imgt:
    
    input2=binangle(io.imread(file))   
    #comp[0]=np.linalg.norm(input1-input2)
    for i in range(576):
          comp[k]+=(input1[i]-input2[i])**2  

    comp[k]=math.sqrt(comp[k])
    
    k+=1

 
find=np.argsort(comp)
# print(find)
for ind in find[0:9]:
    plt.figure()
    io.imshow(imgt[ind])
    save=io.imread(imgt[ind])    
    io.imsave(imgt[ind],save)
    
io.show()

                
                      