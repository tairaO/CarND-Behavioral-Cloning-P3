# -*- coding: utf-8 -*-
"""
Created on Tue May 15 18:44:02 2018

@author: taira
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.models import Sequential, Model, load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import model_to_dot, plot_model
from IPython.display import SVG
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


#%%

### define generator but unused
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0].split('/')[-1]
                center_image = plt.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

### read driving_log.csv
lines = []
with open('C:/Users/taira/Desktop/CarNDL3_data_mouse/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        lines.append(line)

##remove 75% of straight data
lines_removedstraight = []
keep_prob_straight = 0.20
keep_straight_threshold = 0.20

for line in lines:
    if abs(float(line[3]))<= keep_straight_threshold:
        if random.random()<= keep_prob_straight:
            lines_removedstraight.append(line)
    else:
        lines_removedstraight.append(line)

#divide data to training set and validation set
train_samples, validation_samples = train_test_split(lines_removedstraight, test_size=0.2)

'''
### load images and steering data     
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = filename
    image = plt.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

### make augumented data
# 1. remove 75 % of straight driving 
# 2. flip images and steering data 
augument_images = []
augument_measurements = []
keep_prob_straight = 0.20
keep_straight_threshold = 0.15

for image, measurement in zip(images, measurements):
    if abs(measurement) <= keep_straight_threshold:
        if random.random()<= keep_prob_straight:
            augument_images.append(image)
            augument_measurements.append(measurement)
            augument_images.append(np.fliplr(image))
            augument_measurements.append(-measurement)
    else:
        augument_images.append(image)
        augument_measurements.append(measurement)
        augument_images.append(np.fliplr(image))
        augument_measurements.append(-measurement)
# transform list to ndarray
X_train = np.array(augument_images)
y_train = np.array(augument_measurements)
# plot steering distribution
plt.figure()
plt.hist(y_train, bins=41)
plt.xlabel('Steering control [-]')
plt.ylabel('Num [-]')
plt.show()

images.clear()
'''
#%%
###
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

### define model
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3))) #normalize
model.add(Cropping2D(cropping=((70,25),(0,0)))) # crop upper and lower part of image

######### Nvidia architecture + dropout ##############
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(48, 5, 5, subsample=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
###############################3#####
### train model
model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1, callbacks=[early_stopping])
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples), \
                                     validation_data=validation_generator, validation_steps=len(validation_samples), epochs=2, verbose = 1)

### 
plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')



#%%

# display model architecture
plot_model(model, to_file='model.png',show_shapes=True)

#SVG(model_to_dot(model).create(prog='dot', format='svg'))


#%%
# save flipped figure
#plt.imshow(augument_images[1000])
#plt.imshow(augument_images[1001])