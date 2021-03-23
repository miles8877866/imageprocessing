# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
data_x = np.load('./non_linear_x.npy',encoding = "latin1") #加載文件
data_y = np.load('./non_linear_y.npy',encoding = "latin1") #加載文件
data_x = data_x.reshape(100, 1)
data_y = data_y.reshape(100, 1)
w = tf.Variable(-5.0)
b = tf.Variable(2.0)

plt.figure('Draw')
plt.scatter(data_x, data_y)

def Run(input_mat, in_size, out_size):
    W = tf.random.normal(shape = [in_size, out_size])
    b = tf.random.uniform(shape = [out_size])
    outputs = tf.matmul(input_mat, W) + b
    # outputs = Sigmoid(outputs)
    
    return outputs

def Sigmoid(x):
    sig = 1/(1 + np.exp(-x)) 
    
    return sig

def fun(x):
    li = Run(data_x, 1, 10)
    predict = Run(li, 10, 1)
    return predict

# loss = tf.reduce_sum(tf.pow(tf.subtract(data_y,tf.reduce_sum(predict)), 2)) / 100
# plt.scatter(data_x, predict)

# initial graph
plt.ion()
plt.show()

step_amount = 50
lr = 1.002
init_lr = lr #給下面的lr用的

def visual(x, dy_dx, lr, step):
    plt.gca().cla()
    plt.title("step=%03i, x=%03f, dx_dy=%0.3f, lr=%0.3f"%(step, x.numpy(), dy_dx.numpy(), lr), fontsize=18)
    dash_line = np.linspace(0, 6, 100)
    plt.plot(dash_line, fun(dash_line), "--", alpha = 0.5)
    
    plt.scatter(x, fun(x), c="r")
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.xlabel("x", fontsize=18)
    plt.ylabel("y", fontsize=18)
    if step==0:
        plt.pause(1)
    else:
        plt.pause(0.1)

def train_step(x, step):
    
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(tf.pow(tf.subtract(data_y,tf.reduce_sum(data_x*w+b)), 2)) / 100
    dy_dx = tape.gradient(y, w)    
    # lr -= init_lr/step_amount ##若要將lr下降，此行註解拿掉
    visual(x, dy_dx, lr, step) ##視覺化
    x.assign_sub(dy_dx * lr) ##梯度下降法 更新x

x = fun(data_x)
for step in range(step_amount):
    train_step(x, step)
    