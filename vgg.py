# -*- coding: utf-8 -*-
import os
import glob
import tensorflow as tf
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Activation, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import get_file
from keras.preprocessing.image import  img_to_array, load_img
from PIL import Image
import matplotlib.image as mpimg

WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHT_PATH, cache_subdir='models')

model_path = 'vgg.h5'
def load_data(path):
    print("loading data...")
    data = []
    labels = []
    for label in os.listdir(path):
        for r, _, f in os.walk(path+label):
            for img in f:
                data.append(mpimg.imread(path+label+'/'+img))
                labels.append(int(label))
    return np.array(data), np.array(labels)

train_data, train_label = load_data(r'C:/Users/as722/Desktop/tests/pca_poker_data/training/')
data = np.zeros([52, 96, 71, 3])

for i in range(3):
    data[:,:,:, i] = train_data
label = np.eye(4)[train_label]                
#建立神經網路
model = Sequential()
model.add(Conv2D(input_shape=[96 ,71, 3], name = "block1_conv1", filters=64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))
model.add(Conv2D(name = "block1_conv2", filters=64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))
model.add(MaxPooling2D(name = "block1_pool", pool_size=(2, 2), strides=(2, 2)) )    

model.add(Conv2D(name = "block2_conv1", filters=128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))     
model.add(Conv2D(name = "block2_conv2", filters=128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu')) 
model.add(MaxPooling2D(name = "block2_pool", pool_size=(2, 2), strides=(2, 2)) )    

model.add(Conv2D(name = "block3_conv1", filters=256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu')) 
model.add(Conv2D(name = "block3_conv2", filters=256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu')) 
model.add(Conv2D(name = "block3_conv3", filters=256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu')) 
model.add(Conv2D(name = "block3_conv4", filters=256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu')) 
model.add(Conv2D(name = "block3_conv5", filters=256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))      
model.add(MaxPooling2D(name = "block3_pool", pool_size=(2, 2), strides=(2, 2)) ) 
 
model.add(Conv2D(name = "block4_conv1", filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))      
model.add(Conv2D(name = "block4_conv2", filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))      
model.add(Conv2D(name = "block4_conv3", filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))      
model.add(Conv2D(name = "block4_conv4", filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))      
model.add(Conv2D(name = "block4_conv5", filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))      
model.add(MaxPooling2D(name = "block4_pool", pool_size=(2, 2), strides=(2, 2)) )
          
model.add(Conv2D(name = "block5_conv1", filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))      
model.add(Conv2D(name = "block5_conv2", filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))      
model.add(Conv2D(name = "block5_conv3", filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))      
model.add(Conv2D(name = "block5_conv4", filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))      
model.add(Conv2D(name = "block5_conv5", filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())   
model.add(Activation('relu'))      
model.add(MaxPooling2D(name = "block5_pool", pool_size=(2, 2), strides=(2, 2)))

for layer in model.layers:
    layer.trainable = False 

print(layer.trainable)
model.add(Flatten())
model.add(Dense(4096))
model.add(BatchNormalization())   
model.add(Activation('relu'))    
model.add(Dense(1000))
model.add(BatchNormalization())   
model.add(Activation('relu'))    
model.add(Dense(4))
model.add(BatchNormalization())    
model.add(Activation('softmax'))

# for layer in model.layers:
#     layer.trainable = False 

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint("./vgg.h5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
]

# model.load_weights(filepath, by_name = True)
model.load_weights(model_path)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# a = model.fit(data, label, batch_size=4, epochs=100, shuffle = True, callbacks = my_callbacks)
predict = model.evaluate(data, label)
print("%s: %.2f%%" % (model.metrics_names[1], predict[1]*100))
