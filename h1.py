# -*- coding: utf-8 -*-

from skimage import io
import numpy as np
import matplotlib.pyplot as plt

lena = io.imread('lena_24bit.bmp')
h, w, depth = lena.shape


def RGBToHSV(r, g, b):
    
    MAX = max(r, g, b)
    MIN = min(r, g, b)
    
    if MAX == MIN:
        h = 0
    elif MAX == r and g>=b:
        h = 60 * (g-b) / (MAX - MIN) + 0
    elif MAX == r and g<b:
        h = 60 * (g-b) / (MAX - MIN) + 360
    elif MAX == g:
        h = 60 * (g-b) / (MAX - MIN) + 120
    elif MAX == b:
        h = 60 * (g-b) / (MAX - MIN) + 240
    
    if MAX == 0:
        s = 0
    else:
        s = (MAX - MIN)/MAX
    
    v = MAX
    HSV = np.array([h, s, v])
    return HSV



for i in range(h):
    for j in range(w):
        lena[i][j] = RGBToHSV(lena[i][j][0],lena[i][j][1], lena[i][j][2])
                
