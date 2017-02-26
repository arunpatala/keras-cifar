from __future__ import print_function

import numpy as np
import warnings

from keras.layers import merge, Input, Lambda
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
#from imagenet_utils import decode_predictions, preprocess_input

def conv_bn_relu(filter_size, kernel_size=3, bn=True, relu=True, strides=(1,1), border_mode='same'):
    name = str(filter_size)+":"+str(kernel_size)+"x"+str(kernel_size)
    def tmp(input_tensor): 
        x = Convolution2D(filter_size, kernel_size, kernel_size, border_mode=border_mode, subsample=strides)(input_tensor)
        if(bn): x = BatchNormalization()(x)
        if(relu): x = Activation('relu')(x)
        return x
    return tmp

def res_bottleneck(filters, kernel_size=3, strides=(1,1), branch=1,
                   conv_in_shortcut=False, conv=conv_bn_relu):
    def tmp(input_tensor):
            if(len(filters)==1):
                nb_filter1, nb_filter2, nb_filter3 = filters[0], filters[0], filters[0]*4
            else:
                nb_filter1, nb_filter2, nb_filter3 = filters[0], filters[1], filters[2]
            xx = []
            for _ in range(depth):
                x = conv(nb_filter1, 1, strides=strides)(input_tensor)
                x = conv(nb_filter2, kernel_size)(x)
                x = conv(nb_filter3, 1, relu=False)(x)
                xx.append(x)
            if(conv_in_shortcut): 
                input_tensor=conv(nb_filter3, 1, strides=strides, relu=False)(input_tensor)
            xx.append(input_tensor)
            x = merge(xx, mode='sum')
            x = Activation('relu')(x)
            return x
    return tmp

def res(filters, kernel_size=3, strides=(1,1), branch = 1,
        conv_in_shortcut=False, conv=conv_bn_relu):
    def tmp(input_tensor):
            nb_filter = filters[0]
            xx = []
            for _ in range(branch):
                x = conv(nb_filter, kernel_size, strides=strides)(input_tensor)
                x = conv(nb_filter, kernel_size, relu=False)(x)
                xx.append(x)
            if(conv_in_shortcut): 
                input_tensor=conv(nb_filter, 1, strides=strides, relu=False)(input_tensor)
            xx.append(input_tensor)
            x = merge(xx, mode='sum')
            x = Activation('relu')(x)
            return x
    return tmp

def asum(x1,x2):
    def mul(xa):
        return xa[0] * xa[1]
    def mul_shape(xa_shape):
        return xa_shape[0]
    alpha = Lambda(lambda x:K.random_uniform((K.shape(x)[0],1,1,1), 0, 1))(x1)
    xa1 = merge([x1,alpha], mode=mul, output_shape=mul_shape)
    xa2 = merge([x2,alpha], mode=mul, output_shape=mul_shape)
    xa12 = merge([xa1,xa2], mode='sum')
    return xa12

def shakeres(filters, kernel_size=3, strides=(1,1), branch = 2,
        conv_in_shortcut=False, conv=conv_bn_relu):
    assert(branch==2)
    def tmp(input_tensor):
            nb_filter = filters[0]
            x1 = conv(nb_filter, kernel_size, strides=strides)(input_tensor)
            x1 = conv(nb_filter, kernel_size, relu=False)(x1)
            x2 = conv(nb_filter, kernel_size, strides=strides)(input_tensor)
            x2 = conv(nb_filter, kernel_size, relu=False)(x2)
            #alpha = K.random_uniform(K.shape(x1),0,1)
            if(conv_in_shortcut): 
                input_tensor=conv(nb_filter, 1, strides=strides, relu=False)(input_tensor)
            x = merge([input_tensor,asum(x1,x2)], mode='sum')
            x = Activation('relu')(x)
            return x
    return tmp


def block(filters, depth, strides=(2,2), kernel_size=3, branch=1,
          res=res, conv=conv_bn_relu):
    def tmp(input_tensor):
        x = res(filters, kernel_size,  strides=strides, conv_in_shortcut=True, branch=branch, conv=conv)(input_tensor)
        for _ in range(depth-1):
            x = res(filters, kernel_size, conv=conv, branch=branch)(x)
        return x
    return tmp

def blocks(classes, depths, filters=[64,128,256,512], kernel_size=3, branch=1,
           res=res, conv=conv_bn_relu):
    def tmp(input_tensor):
        x = block( [filters[0]], depths[0], kernel_size=kernel_size, strides=(1, 1), branch=branch,
                 res=res, conv=conv)(input_tensor)
        for i in range(1,len(depths)):
            x = block([filters[i]], depths[i], strides=(2, 2), kernel_size=kernel_size, branch=branch,
                     res=res, conv=conv)(x)
        _,_,H,W = x.get_shape()
        H,W = H.value, W.value
        x = AveragePooling2D((H, W), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(classes, activation='softmax')(x)
        return x
    return tmp

def imagenet():
    classes = 1000
    input_shape = (3, 224, 224)
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)
    x = conv(x, 64, 7, strides=(2,2),border_mode='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = blocks(x, classes, [3,4,6,3])
    return Model(img_input, x)

def cifar(depth,widenFactor=1):   
    assert(((depth - 2) % 6) == 0)
    n = int((depth - 2) / 6)
    classes = 10
    input_shape = (3, 32, 32)
    print(' | ResNet-' + str(depth) + ' CIFAR-10')
    img_input = Input(shape=input_shape)
    x = blocks(classes, [n,n,n],[16,32*widenFactor,64*widenFactor],branch=2,res=shakeres)(img_input)
    return Model(img_input, x)