# -*- coding: utf-8 -*-
from skimage import io
import math
import numpy as np
import matplotlib.pyplot as plt

sobel1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
lena = io.imread('lena512_8bit.bmp')
h, w = lena.shape
padd = np.zeros((h+2, w+2),dtype=np.uint8)
fin1 = np.zeros((h, w),dtype=np.uint8)
for i in range(1, h+1):
    for j in range(1, w+1):
        padd[i][j] = lena[i-1][j-1]
        
padd[0,:] = padd[1,:]
padd[h+1, :] = padd[h, :]
padd[:, 0] = padd[:, 1]
padd[:, w+1] = padd[:, w-1]

for i in range(1, h+1):
    for j in range(1, h+1):
        fin1[i-1,j-1] = math.sqrt(((sobel1*padd[i-1:i+2, j-1:j+2]).sum())**2+((sobel2*padd[i-1:i+2, j-1:j+2]).sum())**2)   

io.imshow(fin1)
plt.figure()
io.show()
io.imsave("lena_edge.bmp", fin1)