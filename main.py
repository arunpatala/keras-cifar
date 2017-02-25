from __future__ import print_function

import numpy as np
import warnings

from keras.layers import merge, Input
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

def cifar(depth):   
    assert(((depth - 2) % 6) == 0)
    n = int((depth - 2) / 6)
    classes = 10
    input_shape = (3, 32, 32)
    print(' | ResNet-' + str(depth) + ' CIFAR-10')
    img_input = Input(shape=input_shape)
    x = blocks(classes, [n,n,n],[16,32,64],branch=2)(img_input)
    return Model(img_input, x)
model = cifar(20)
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
data_augmentation = True
if not data_augmentation:

	print('Not using data augmentation.')
	model.fit(X_train, Y_train,
		  batch_size=batch_size,
		  nb_epoch=10,
		  validation_data=(X_test, Y_test),
		  shuffle=True)
else:
	print('Using real-time data augmentation.')
	# This will do preprocessing and realtime data augmentation:
	datagen = ImageDataGenerator(
	featurewise_center=False,  # set input mean to 0 over the dataset
	samplewise_center=False,  # set each sample mean to 0
	featurewise_std_normalization=False,  # divide inputs by std of the dataset
	samplewise_std_normalization=False,  # divide each input by its std
	zca_whitening=False,  # apply ZCA whitening
	rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
	width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
	height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
	horizontal_flip=True,  # randomly flip images
	vertical_flip=False)  # randomly flip images

	# Compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(X_train)

	# Fit the model on the batches generated by datagen.flow().
	model.fit_generator(datagen.flow(X_train, Y_train,
				     batch_size=batch_size),
			samples_per_epoch=X_train.shape[0],
			nb_epoch=nb_epoch,
			validation_data=(X_test, Y_test))

