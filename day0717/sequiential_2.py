# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D

x = np.load('./non_linear_x.npy')
y = np.load('./non_linear_y.npy')
x = x.reshape((-1, 1))
y = y.reshape((-1, 1))

model = Sequential([Dense(10, activation="sigmoid"), Dense(1)])
model.compile(optimizer='sgd', loss='mse')
model.fit(x, y, batch_size=100, epochs=5000)

predict=model(x)
plt.scatter(x,predict,c="r")
plt.scatter(x,y)
    
plt.xlim(0,6)
plt.ylim(0,10)
plt.xlabel("x",fontsize=18)
plt.ylabel("y",fontsize=18)
