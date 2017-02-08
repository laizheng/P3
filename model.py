from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import os

import tensorflow as tf
import matplotlib.gridspec as gridspec
import random
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from scipy import signal
import scipy.stats as stats
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import model_from_json

#np.random.seed(1337)
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape, BatchNormalization, Lambda, SpatialDropout2D
from keras.layers.advanced_activations import ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', '/', "data_path (/home/ubuntu/pynb/bc/data/)")

def imgPreprocess(img):
    img = cv2.resize(img[60:140],(200,66))
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img

def addNormalData(d,data_path):
    """
    :return: the paths to the files(X_path) and target(y). The returns are lists not np.array.
    """
    X_path = []
    y = []
    for index, row in d.iterrows():
        X_path.append(data_path+'norm/'+(row["center"]))
        y.append(row["steering"])
        X_path.append(data_path+'norm/'+(row["left"]))
        y.append(row["steering"]+0.15)
        X_path.append(data_path+'norm/'+(row["right"]))
        y.append(row["steering"]-0.15)
    print("Total Training Examples for normal data:")
    print(len(y))
    return X_path, y

def addRecoveryL2RData(d,data_path):
    """
    :return: the paths to the files(X_path) and target(y). The returns are lists not np.array.
    """
    X_path = []
    y = []
    for index, row in d.iterrows():
        X_path.append(data_path+'recovery_from_l2r/'+(row["center"]))
        y.append(row["steering"])
    print("Total Training Examples for L2R data:")
    print(len(y))
    return X_path, y

def generate_from_directory(X_path, y, batch_size = 32, pr_threshold = 0.3):
    """
    :param X_path: paths to stored images (list)
    :param y: target angles (list)
    :return: image in the form of numpy arrays; target values in the form of np.array
    """
    assert len(X_path) == len(y)
    y = np.vstack(y)
    numOfBatches = int(len(y)/batch_size)
    start_index = 0
    while 1:
        X_ret = []
        y_ret = []
        j=0
        while j<batch_size:
            #print("start_index={}".format(start_index))
            if abs(y[start_index])<0.05:
                pr_val = np.random.uniform()
                if pr_val>pr_threshold:
                    #print("pr_val={},pr_threshold={}".format(pr_val,pr_threshold))
                    start_index += 1
                    if start_index == len(y):
                        start_index = 0
                    continue
            x = mpimg.imread(X_path[start_index])
            x = imgPreprocess(x)
            if len(x.shape)<3:
                x = np.resize(x,(x.shape[0],x.shape[1],1))
            X_ret.append(x)
            y_ret.append(y[start_index])
            start_index += 1
            if start_index == len(y):
                start_index = 0
            j += 1
            #print("j={}".format(j))
        X_ret = np.array(X_ret)
        y_ret = np.array(y_ret)
        yield (X_ret, y_ret)


def addRecoveryR2LData(d,data_path):
    """
    :return: the paths to the files(X_path) and target(y). The returns are lists not np.array.
    """
    X_path = []
    y = []
    for index, row in d.iterrows():
        X_path.append(data_path+'recovery_from_r2l/'+(row["center"]))
        y.append(row["steering"])
    print("Total Training Examples for R2L data:")
    print(len(y))
    return X_path, y

def getModel(input_shape):
    # number of convolutional filters to use
    nb_filters_conv1 = 24
    nb_filters_conv2 = 36
    nb_filters_conv3 = 48
    nb_filters_conv4 = 64
    nb_filters_conv5 = 64
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size_5 = (5, 5)
    kernel_size_3 = (3, 3)

    model = Sequential()
    #model.add(Lambda(lambda x: (x - 128.0) / 255.0,input_shape=input_shape, name='normalization'))
    model.add(BatchNormalization(name='bn1',input_shape=input_shape))
    model.add(Convolution2D(nb_filters_conv1, kernel_size_5[0], kernel_size_5[1],
                            #init = 'glorot_normal',
                            #init = my_init,
                            subsample=(2, 2),
                            border_mode='valid',
                            name="conv1"))
    model.add(BatchNormalization(name='bn2'))
    model.add(ELU(name='act1'))
    #model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters_conv2, kernel_size_5[0], kernel_size_5[1],
                            #init = 'glorot_normal',
                            #init = my_init,
                            subsample=(2, 2),
                            border_mode='valid',
                            name="conv2"))
    model.add(BatchNormalization(name='bn3'))
    model.add(ELU(name='act2'))
    model.add(Convolution2D(nb_filters_conv3, kernel_size_5[0], kernel_size_5[1],
                            #init = 'glorot_normal',
                            #init = my_init,
                            subsample=(2, 2),
                            border_mode='valid',
                            name="conv3"))
    model.add(BatchNormalization(name='bn4'))
    model.add(ELU(name='act3'))
    model.add(Convolution2D(nb_filters_conv4, kernel_size_3[0], kernel_size_3[1],
                            #init = 'glorot_normal',
                            #init = my_init,
                            subsample=(1, 1),
                            border_mode='valid',
                            name="conv4"))
    model.add(BatchNormalization(name='bn5'))
    model.add(ELU(name='act4'))
    model.add(Convolution2D(nb_filters_conv5, kernel_size_3[0], kernel_size_3[1],
                            #init = 'glorot_normal',
                            #init = my_init,
                            subsample=(1, 1),
                            border_mode='valid',
                            name="conv5"))
    model.add(BatchNormalization(name='bn6'))
    model.add(ELU(name='act5'))
    model.add(Flatten(name="flat"))
    #model.add(Dense(500, name = "FC0"))#,init = my_init))
    #model.add(Activation('relu'))
    model.add(Dense(100, name = "FC1"))#,init = my_init))
    model.add(BatchNormalization(name='bn7'))
    model.add(ELU(name='act6'))
    model.add(Dense(50, name = "FC2"))#,init = my_init))
    model.add(BatchNormalization(name='bn8'))
    model.add(ELU(name='act7'))
    model.add(Dense(10, name = "FC3"))#,init = my_init))
    model.add(Dense(1, name = "output"))
    #model.add(Activation('tanh'))
    return model

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        print('BEGIN TRAINING')
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def main(_):
    names = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    data_path = FLAGS.data_path
    d_norm = pd.read_csv(data_path+"/norm/driving_log.csv", sep=",\s*", engine='python', names=names)
    d_l2r = pd.read_csv(data_path+"/recovery_from_l2r/driving_log.csv", sep=",\s*", engine='python', names=names)
    d_r2l = pd.read_csv(data_path+"/recovery_from_r2l/driving_log.csv", sep=",\s*", engine='python', names=names)
    X_path_normal, y_normal = addNormalData(d_norm,data_path)
    X_path_left_to_right, y_left_to_right = addRecoveryL2RData(d_l2r,data_path)
    X_path_right_to_left, y_right_to_left = addRecoveryR2LData(d_r2l,data_path)
    X_path = X_path_normal + X_path_left_to_right + X_path_right_to_left
    y = y_normal + y_left_to_right + y_right_to_left
    X_train_path, X_val_path, y_train, y_val = train_test_split(X_path, y, test_size=0.1, random_state=42)
    X_norm_train_path, X_norm_val_path, y_norm_train, y_norm_val = \
        train_test_split(X_path_normal, y_normal, test_size=0.1, random_state=42)
    input_shape = imgPreprocess(mpimg.imread(X_path[0])).shape
    if len(input_shape) < 3:
        input_shape = input_shape + (1,)
    model = getModel(input_shape)
    opt = Adam()
    # opt = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=opt)  # ,metrics=['accuracy'])
    model.summary()

    ### Train Model
    batch_history = LossHistory()
    batch_size = 128
    X_train_feed = X_norm_train_path
    y_train_feed = y_norm_train
    X_val_feed = X_norm_val_path
    y_val_feed = y_norm_val
    samples_per_epoch = (int(len(y_train_feed) / float(batch_size)) + 1) * batch_size
    nb_val_samples = (int(len(y_val_feed) / float(batch_size)) + 1) * batch_size
    history = model.fit_generator(
        generate_from_directory(X_train_feed, y_train_feed, batch_size=batch_size, pr_threshold=0.50),
        samples_per_epoch=samples_per_epoch, nb_epoch=4,
        validation_data=generate_from_directory(X_val_feed, y_val_feed, batch_size=batch_size, pr_threshold=1.1),
        nb_val_samples=nb_val_samples, \
        callbacks=[batch_history])
    import json
    json_string = model.to_json()
    with open('model.json', 'w') as f:
        json.dump(json_string, f)
    model.save_weights('model.h5')

if __name__ == '__main__':
    tf.app.run()