# -*- coding: utf-8 -*-
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


step_amount = 800
lr = 0.1
lr_1 = 0.5
initial_lr =lr

x = np.load('./non_linear_x.npy')
y = np.load('./non_linear_y.npy')

###宣告模型
model = Sequential([Dense(10, activation = "sigmoid"), Dense(1)] )


x = x.reshape((-1, 1))
y = y.reshape((-1, 1))
model(x)
print(model.summary())
n = x.size
print(model.layers[0].weights[0])

# W =tf.Variable(tf.random.uniform(shape=[1,10]))
# W1 =tf.Variable(tf.random.uniform(shape=[10,1]))

# B = tf.Variable(tf.random.uniform(shape=[10]))
# B1 =tf.Variable(tf.random.uniform(shape=[1]))


plt.ion()
def visual(predict,step,loss,dw,db,lr):
    plt.gca().cla()
    plt.title(" step=%0.3f,lr=%0.3f"%(step,lr),fontsize=10)
    
    plt.scatter(x,predict,c="r")
    plt.scatter(x,y)
    
    plt.xlim(0,6)
    plt.ylim(0,10)
    plt.xlabel("x",fontsize=18)
    plt.ylabel("y",fontsize=18)

    if(step==0):plt.pause(1)
    else:plt.pause(0.1)
    

# def sigmoid(x):
#     x=tf.divide(1,(1+tf.exp(x)))
#     return x

# def RUN(X,W,B,W1,B1):

#     X=tf.add(tf.matmul(X,W),B)
#     X=sigmoid(X)
#     X=tf.add(tf.matmul(X,W1),B1)
    
#     return X

def train_step(step):
    with tf.GradientTape(persistent=True) as tape:
    
          predict=model(x)
          loss = tf.reduce_sum(tf.square(y-predict))/n
         
         
    dw_dloss = tape.gradient(loss, model.layers[0].weights[0])
    db_dloss = tape.gradient(loss, model.layers[0].weights[1]) 
    dw1_dloss =tape.gradient(loss, model.layers[1].weights[0])
    db1_dloss =tape.gradient(loss, model.layers[1].weights[1]) 
    
    visual(predict,step,loss,dw_dloss,db_dloss,lr)
    
    #plt.scatter(x,predict,c="r")
    #plt.scatter(x,y) 
    model.layers[0].weights[0].assign_sub(dw_dloss * lr)    
    model.layers[0].weights[1].assign_sub(db_dloss * lr)
    model.layers[1].weights[0].assign_sub(dw1_dloss * lr)    
    model.layers[1].weights[1].assign_sub(db1_dloss * lr)
#    print(w.numpy(),b.numpy())


# for step in range(step_amount):
#     train_step(step)
#     lr -= initial_lr/step_amount
#     print(step)
    

plt.show()




