# -*- coding: utf-8 -*-
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
import cv2 
plt.rcParams['font.sans-serif']=['SimSun'] #設定字型,以正常顯示中文
plt.rcParams['axes.unicode_minus']=False #用來正常顯示 負號

img = cv2.imread('flower_24bit.bmp')
h, w, depth = img.shape
(b, g, r) = cv2.split(img)
#轉灰色
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hist_gray = np.zeros([256])
mag_gray = np.zeros([256])#強度
red_list = np.zeros([256])
red = np.zeros([h, w], dtype=np.uint8)
green_list = np.zeros([256])
blue_list = np.zeros([256])

for i in range(256):
    gray_value = gray_img[i]
    hist_gray[gray_value]+=1
    mag_gray[gray_value] += gray_value 
###normalize
for i in range(256):
    if mag_gray[i]==0:
        hist_gray[i] = hist_gray[i]
    else:
        hist_gray[i] = hist_gray[i] / mag_gray[i]
     

merged_b = cv2.merge([b,g*0,r*0])#合併r、g、b分量
merged_g = cv2.merge([b*0,g,r*0])#合併r、g、b分量
merged_r = cv2.merge([b*0,g*0,r])#合併r、g、b分量

merge_img = cv2.merge([r, g, b])
###count red list
for i in range(h):
    for j in range(w):
        red_list[r[i,j]]+=1
###count green list
for i in range(h):
    for j in range(w):
        value = merged_g[i, j, 1]
        green_list[value]+=1
###count blue
for i in range(256):
    for i in range(h):
        for j in range(w):
            blue_list[b[i,j]]+=1

for i in range(256):
        blue_list[i] = blue_list[i]+blue_list[i-1]
blue_list = blue_list / blue_list.sum()

###count green list
for i in range(h):
    for j in range(w):
        value = merged_g[i, j, 1]
        green_list[value]+=1
        
fig3 = plt.figure(constrained_layout=True)
gs = fig3.add_gridspec(3, 3)

f3_ax1 = fig3.add_subplot(gs[0:2, 0:2])
img=img.astype(np.uint8)
f3_ax1.imshow(merge_img)
f3_ax1.get_xaxis().set_visible(False)
f3_ax1.get_yaxis().set_visible(False)
f3_ax1.set_title('進')

f3_ax2 = fig3.add_subplot(gs[2, 0:2])
f3_ax2.set_title('hist')
f3_ax2.hist(gray_img.ravel(), bins = range(256), density = 1)

f3_ax3 = fig3.add_subplot(gs[0, -1])
f3_ax3.set_title('R')
f3_ax3.plot(red_list, color='red')

f3_ax4 = fig3.add_subplot(gs[1, -1])
f3_ax4.set_title('g')
f3_ax4.bar(np.arange(256), green_list , color = 'green')

f3_ax5 = fig3.add_subplot(gs[2, -1])
f3_ax5.set_title('b')
f3_ax5.plot(blue_list, color = 'blue')
f3_ax5.grid(1)
# cv2.imshow('img', img)
# cv2.imshow("Gray",gray_img)
# cv2.imshow('b.bmp', merged_b)
# cv2.imshow('g.bmp', merged_g)
# cv2.imshow('r.bmp', merged_r)

# cv2.waitKey()
# cv2.destroyAllWindows()