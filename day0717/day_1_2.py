# -*- coding: utf-8 -*-
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

w=tf.Variable(-5.0)
b=tf.Variable(2.0)

x=np.load('./x_data.npy')

y=np.load('./y_data.npy')



n=x.size

def fun(x,w,b):
    return tf.add(tf.multiply(x,w),b)

step_amount =25
lr=0.2
init_lr =lr

plt.ion()
def visual(w,b, predict,step,loss,dw,db):
    plt.gca().cla()
    plt.title("step=%03i, w=%0.3f, b=%0.3f, lr=%0.3f,loss=%0.3f,dw=%0.3f,db=%0.3f"%(step,w.numpy(),b.numpy(),lr,loss,dw,db),fontsize=10)
    
    dash_line=np.linspace(0,6,100)
    plt.plot(dash_line,fun(dash_line,w,b),"--",alpha=0.5)
    plt.scatter(x,predict,c="r")
    plt.scatter(x,y)
    
    plt.xlim(-3,3)
    plt.ylim(-15,15)
    plt.xlabel("x",fontsize=18)
    plt.ylabel("y",fontsize=18)

    if(step==0):plt.pause(1)
    else:plt.pause(0.1)
    

def train_step(w,b,step):

    with tf.GradientTape(persistent=True) as tape:
        #x*w+b
        predict=tf.add(tf.multiply(x,w),b)
        
        loss = tf.reduce_sum(tf.square(y-predict))/n
            
    dw_dloss =tape.gradient(loss,w)
    db_dloss =tape.gradient(loss,b) 
    
    visual(w,b,predict,step,loss,dw_dloss,db_dloss)
    
    #plt.scatter(x,predict,c="r")
    #plt.scatter(x,y)    
    w.assign_sub(dw_dloss * lr)    
    b.assign_sub(db_dloss * lr)
    print(w.numpy(),b.numpy())



for step in range(step_amount):
    train_step(w,b,step)
    
    print(step)
    

plt.show()