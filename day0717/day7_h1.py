# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
data_x = np.load('./x_data.npy',encoding = "latin1") #加載文件
data_y = np.load('./y_data.npy',encoding = "latin1") #加載文件

plt.figure('Draw')
plt.scatter(data_x, data_y)

w = tf.Variable(-5.0)
b = tf.Variable(2.0)

predict = np.zeros([100])
for i in range(0, 100):
    predict[i] = data_x[i] * w + b
    
with tf.GradientTape(persistent = True) as tape:
    loss = tf.reduce_sum(tf.pow(tf.subtract(data_y,data_x*w+b), 2)) / 100
dw = tape.gradient(loss, w)
db = tape.gradient(loss, b)

plt.title("w = %.2f b = %.2f dw = %.2f db = %.2f" %(w.numpy(), b.numpy(), dw.numpy(), db.numpy()))
plt.scatter(data_x, predict)
    
plt.show()