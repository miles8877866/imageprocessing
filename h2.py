# -*- coding: utf-8 -*-
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

img = io.imread('test_rgb.jpg')
h, w, depth = img.shape
map_skin = np.zeros([h, w], np.bool)
arr_skin = np.zeros([h, w, depth], np.uint8)
arr_skin2 = np.zeros([h, w, depth], np.uint8)

def isSkin(p, colorspace='YCbCr'):
    #use dictionary 來定義可判斷膚色的顏色空間及膚色範圍:YCbCr, HSV
    skin_color = { "YCbCr" :(np.array([80, 85, 135]), np.array([255, 135, 180])),
                   "HSV" :(np.array([0, 0.23, 0]), np.array([50/360.0, 0.68, 1])),
                   "YCbCr_Asia" :(np.array([0, 0, 140]), np.array([255, 255, 160])),
                   "YCbCr2" :(np.array([0, 77, 133]), np.array([255, 127, 173])),
                   "RGB" :(np.array([150, 50, 20]), np.array([255, 255, 255]))
                  }
    
    if colorspace not in skin_color.keys():
        raise ValueError("Color Space Error!!")
    
    if p.size != 3:
        raise ValueError("Pixel value depth error!!")
        
    result1 = skin_color[colorspace][0] <= p
    result2 = p <= skin_color[colorspace][1]
    
    if colorspace == 'RGB':
        r, g, b = p[:3]
        result = r>g and r>b and r-g>15
        return result and result1.all() and result2.all()
    
    return result1.all() and result2.all()

for i in range(h):
    for j in range(w):
        if isSkin(img[i, j], 'RGB'):
            map_skin[i, j] = 1
            arr_skin[i, j, :] = img[i, j, :]
            
for i in range(h):
    for j in range(w):
        if map_skin[i, j]==1:
            arr_skin2[i, j, :] = img[i, j, :]
plt.figure()
io.imshow(img)
plt.figure()
io.imshow(map_skin)
plt.figure()
io.imshow(arr_skin2)
io.show()