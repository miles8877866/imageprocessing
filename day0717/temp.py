# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

data_x = np.load('./non_linear_x.npy',encoding = "latin1") #加載文件
data_y = np.load('./non_linear_y.npy',encoding = "latin1") #加載文件
data_x = data_x.reshape(100, 1)
data_y = data_y.reshape(100, 1)

w1_origin = tf.random.normal(shape = [1, 10])
b1_origin = tf.random.uniform(shape = [10])
w2_origin = tf.random.normal(shape = [10, 1])
b2_origin = tf.random.uniform(shape = [1])

###sigmoid
def Sigmoid(x):
    sig = 1/(1 + tf.exp(-x)) 
    
    return sig

# plt.figure('origin')
# plt.scatter(data_x, data_y)
# plt.scatter(data_x, outputs)

w1 = tf.Variable(w1_origin)
b1 = tf.Variable(b1_origin)
w2 = tf.Variable(w2_origin)
b2 = tf.Variable(b2_origin)

step_amount = 50
lr = 1.002
init_lr = lr #給下面的lr用的

def visual(w1, b1,w2, b2, dw1, db1, dw2, db2, lr, step):
    
    
    
    pass
def train_step(step, input_mat, w1, b1, w2, b2):
    
    with tf.GradientTape(persistent = True) as tape:
        outputs = tf.matmul(input_mat, w1) + b1
        outputs = Sigmoid(outputs)
        outputs = tf.matmul(outputs, w2) + b2
        print(outputs)
        loss = tf.reduce_sum(tf.pow(data_y - outputs, 2)) / 100
    
    
    dw1 = tape.gradient(loss, w1)
    db1 = tape.gradient(loss, b1)
    dw2 = tape.gradient(loss, w2)
    db2 = tape.gradient(loss, b2)
    # lr -= init_lr/step_amount ##若要將lr下降，此行註解拿掉
    visual(w1, b1, w2, b2, dw1, db1, dw2, db2, lr, step)
    w1.assign_sub(dw1 * lr)
    b1.assign_sub(db1 * lr)
    w2.assign_sub(dw2 * lr)
    b2.assign_sub(db2 * lr)



for step in range(step_amount):
    train_step(step, data_x, w1, b1, w2, b2)
    print(step)