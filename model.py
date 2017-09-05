#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 06:45:09 2017

@author: subbu
"""

import csv
import cv2
import numpy as np

images = []
measurements = []
#basedirs = ['./data','./data_recovery','./data_recovery2','./data_curve']
basedirs = ['./data']
num_cameras = 1
path_separators = ['/','\\','/','\\']
for index in range(len(basedirs)):
    basedir = basedirs[index]
    lines = []
    with open(basedir +'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in  reader:
            lines.append(line)

    # Strip out the first line
    lines.pop(0)
    for line in lines:
        for cam in range(num_cameras):
        #cam = np.random.choice(range(3))
            sourcepath = line[cam]
            filename = sourcepath.split(path_separators[index])[-1]
            current_path = basedir + '/IMG/'+filename
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mychannels = cv2.split(image)
            image = mychannels[1]
            #image = image[65:135,0:320]
            images.append(image)
            measurement = float(line[3])
            if cam==1:
                measurement += 0.25
            elif cam==2:
                measurement -= 0.25
            measurements.append(measurement)
            images.append(cv2.flip(image,1))
            measurements.append(measurement*-1.0)
    
X_train = np.array(images)
X_train = np.expand_dims(X_train,4)
y_train = np.array(measurements)

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D

newModel = True
if newModel:
    model = Sequential()
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape=(160,320,1)))
    model.add(Convolution2D(6,5,5,activation="relu",dim_ordering='tf'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu",dim_ordering='tf'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    #model.add(Dense(120))
    #model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
else:
    model = load_model('multmodel.h5')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('schannelud.h5')

pmeasurements = []
for index in range(len(measurements)):
    nimage = np.asarray(images[index])
    pmeas = model.predict(nimage[None,:,:,:],batch_size=1)
    pmeasurements.append(pmeas)
