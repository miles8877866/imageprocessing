# -*- coding: utf-8 -*-
###匯入庫
import os
import cv2
###
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
import torch
import math
from collections import OrderedDict
import torch.nn as nn
import torch.nn.init as init
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable
WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHT_PATH, cache_subdir='models')
model_path = 'vgg.h5'


class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = th.cat(pooling_layers, dim=-1)
        return x

class DetectionNetSPP(nn.Module):
    """
    Expected input size is 64x64
    """

    def __init__(self, spp_level=3):
        super(DetectionNetSPP, self).__init__()
        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2**(i*2)
        print(self.num_grids)
            
        self.conv_model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(3, 128, 3)), 
          ('relu1', nn.ReLU()),
          ('pool1', nn.MaxPool2d(2)),
          ('conv2', nn.Conv2d(128, 128, 3)),
          ('relu2', nn.ReLU()),
          ('pool2', nn.MaxPool2d(2)),
          ('conv3', nn.Conv2d(128, 128, 3)), 
          ('relu3', nn.ReLU()),
          ('pool3', nn.MaxPool2d(2)),
          ('conv4', nn.Conv2d(128, 128, 3)),
          ('relu4', nn.ReLU())
        ]))
        
        self.spp_layer = SPPLayer(spp_level)
        
        self.linear_model = nn.Sequential(OrderedDict([
          ('fc1', nn.Linear(self.num_grids*128, 1024)),
          ('fc1_relu', nn.ReLU()),
          ('fc2', nn.Linear(1024, 2)),
        ]))

    def forward(self, x):
        x = self.conv_model(x)
        x = self.spp_layer(x)
        x = self.linear_model(x)
        return x
    
def read_directory(path):
    print("loading img...")
    print(path) #just for test
    data = []
    labels = []
    for label in os.listdir(path):
        for r, _, f in os.walk(path+label):
            for img in f:
                data.append(mpimg.imread(path+label+'/'+img))
                labels.append(int(label))
    return np.array(data), np.array(labels)
        #img is used to store the image data 
        # img = cv2.imread(directory_name + "/" + filename)
        # array_of_img.append(img)
        #print(img)
        # print(array_of_img)
train_data, train_label = read_directory(r"C:/Users/as722/Desktop/im/")
label = np.eye(5)[train_label]                

# #建立神經網路
model = Sequential()
model.add(Conv2D(name = "block1_conv1", filters=64, kernel_size=(3,3), padding='same'))
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

model.add(SPPLayer(train_data))
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

# # for layer in model.layers:
# #     layer.trainable = False 

# my_callbacks = [
#     tf.keras.callbacks.ModelCheckpoint("./vgg.h5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
# ]

# # model.load_weights(filepath, by_name = True)
# model.load_weights(model_path)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, label, batch_size=4, epochs=100, shuffle = True)
predict = model.evaluate(train_data, label)
print("%s: %.2f%%" % (model.metrics_names[1], predict[1]*100))
# # -*- coding: utf-8 -*-
