from skimage import io
import numpy as np
import matplotlib.pyplot as plt

lena = io.imread('grey.bmp')
h, w = lena.shape
pepper = io.imread('lena_salt_pepper.bmp')
h1, w1 = pepper.shape

low = np.array([[1,2,1], [2,4,2], [1,2,1]])

padd = np.zeros((h+2, w+2), dtype=int)
fin1 = np.zeros((h+2, w+2), dtype=int)
padd2 = np.zeros((h1+2, w1+2), dtype=int)
fin2 = np.zeros((h1+2, w1+2), dtype=int)

for i in range(1, h+1):
    for j in range(1, w+1):
        padd[i][j] = lena[i-1][j-1]

for i in range(1, h+1):
    for j in range(1, w+1):
        fin1[i][j] = padd[i][j]

for k in range(10):
        for i in range(1, h+1):
            for j in range(1, h+1):
                fin1[i][j] = (low*padd[i-1:i+2, j-1:j+2]).sum()//16   
        padd = fin1.copy()         

for i in range(1, h1+1):
    for j in range(1, w1+1):
        padd2[i][j] = pepper[i-1][j-1]

for i in range(1, h1+1):
    for j in range(1, w1+1):
        fin2[i][j] = padd2[i][j]

for k in range(10):
        for i in range(1, h1+1):
            for j in range(1, w1+1):
                fin2[i][j] = (low*padd2[i-1:i+2, j-1:j+2]).sum()//16   
        padd2 = fin2.copy()     


io.imshow(fin1)
plt.figure()
io.imshow(fin2)
plt.figure()
io.show()
io.imsave("lena_low.bmp", fin1)
io.imsave("lena_noise_low.bmp", fin2)
        
        