from skimage import io
import numpy as np
import matplotlib.pyplot as plt

lena = io.imread('lena512_8bit.bmp')
h, w = lena.shape
hist = np.zeros((256), dtype=int)
hist2 = np.zeros((256), dtype=int)
#min:25 ,max:245 
for row in range(h):
    for col in range(w):
        gray = lena[row][col]
        hist[gray]+=1
        
for row in range(h):
    for col in range(w):
        lena[row][col] = ((lena[row][col]-25) / (245-25)) * 255

for row in range(h):
    for col in range(w):
        gray2 = lena[row][col]
        hist2[gray2]+=1

io.imshow(lena)
io.show()
# io.imsave('lena_4_3.bmp', lena)
plt.figure()
plt.plot(hist)
plt.show()
plt.figure()
plt.plot(hist2)
plt.show()

