# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:45:12 2019

@author: MUJ
"""

import keras
from keras.layers import MaxPooling2D,Conv2D,Input,Add,Flatten,AveragePooling2D,Dense,BatchNormalization,ZeroPadding2D,Activation
from keras.models import Model

def ConvolutionLayer(x,f,filters,s=2):
    f1,f2,f3 = filters
    x_shortcut = x
    x = Conv2D(filters =f1,kernel_size = (1,1),strides = (s,s))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters =f2,kernel_size = (f,f),strides = (1,1),padding = 'same')(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters =f3,kernel_size = (1,1),strides = (1,1))(x)
    x = BatchNormalization(axis = 3)(x)
    
    x_shortcut = Conv2D(filters =f3,kernel_size = (1,1),strides=(s,s))(x_shortcut)
    x_shortcut = BatchNormalization(axis = 3)(x_shortcut)
    
    x = Add()([x,x_shortcut])
    x = Activation('relu')(x)
    return x
def IdentityBlock(x,f,filters):
    f1,f2,f3 = filters

    x_shortcut = x
    x = Conv2D(f1,(1,1),strides = (1,1))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters =f2,kernel_size = (f,f),strides = (1,1),padding = 'same')(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters = f3,kernel_size = (1,1),strides = (1,1))(x)
    x = BatchNormalization(axis = 3)(x)
    
    x = Add()([x,x_shortcut])
    x = Activation('relu')(x)
    
    return x


def ResNet_50(input_shape = (64,64,3),output_classes=6):
    x_input = Input(input_shape)
    
    x= ZeroPadding2D((3,3))(x_input)
    #stage 1
    x = Conv2D(64,(7,7),strides=(2,2))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3),strides = (2,2))(x)
    
    #stage 2
    x = ConvolutionLayer(x,f=3,filters=[64,64,256],s=1)
    x = IdentityBlock(x,3,[64,64,256])
    x = IdentityBlock(x,3,[64,64,256])
    
    #stage 3
    x = ConvolutionLayer(x,f=3,filters=[128,128,512],s=2)
    x = IdentityBlock(x,3,[128,128,512])
    x = IdentityBlock(x,3,[128,128,512])
    x = IdentityBlock(x,3,[128,128,512])
    
    #stage 4
    x = ConvolutionLayer(x,f=3,filters=[256,256,1024],s=2)
    x = IdentityBlock(x,3,[256,256,1024])
    x = IdentityBlock(x,3,[256,256,1024])
    x = IdentityBlock(x,3,[256,256,1024])
    x = IdentityBlock(x,3,[256,256,1024])
    x = IdentityBlock(x,3,[256,256,1024])
    
    #stage 5
    x = ConvolutionLayer(x,f=3,filters=[512,512,2048],s=2)
    x = IdentityBlock(x,3,[512,512,2048])
    x = IdentityBlock(x,3,[512,512,2048])
    
    x = AveragePooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(output_classes, activation = 'softmax')(x)
    model = Model(inputs = x_input,outputs = x)
    return model

model = ResNet_50(input_shape = (64,64,3), output_classes =6)
model.compile(optimizer = "adam",loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()

 