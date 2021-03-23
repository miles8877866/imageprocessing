# -*- coding: utf-8 -*-
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

pepper = io.imread('lena_salt_pepper.bmp')
h1, w1 = pepper.shape

padd2 = np.zeros((h1+2, w1+2), dtype=int)
fin2 = np.zeros((h1+2, w1+2), dtype=int)

for i in range(1, h1+1):
    for j in range(1, w1+1):
        padd2[i][j] = pepper[i-1][j-1]

for i in range(1, h1+1):
    for j in range(1, w1+1):
        fin2[i][j] = padd2[i][j]

for k in range(1):
        for i in range(1, h1+1):
            for j in range(1, w1+1):
                fin2[i][j] = np.median(padd2[i-1:i+2, j-1:j+2] ) 
        padd2 = fin2.copy()     


io.imshow(fin2.astype("uint8"))
plt.figure()
io.show()

io.imsave("lena_noise_median.bmp", fin2)
        
        