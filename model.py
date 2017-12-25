import os
import csv
from random import shuffle
import numpy as np
import cv2
import sklearn
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DROPOUT_RATE = 0.25

# keras generator for creating raw image data from csv samples
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # for center, left, right indexes
                for i in range(0,3):
                    # get image from csv file name
                    name = './IMG/'+batch_sample[i].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    # alter angle if right or left camera photo
                    # if left
                    if i == 1:
                        center_angle += 0.4
                    # if right
                    elif i == 2:
                        center_angle -= 0.4
                    # create flipped copy
                    image_flipped = np.fliplr(center_image)
                    measurement_flipped = -center_angle
                    # add original and flipped copy to array
                    # convert BGR to RGB
                    images.append(center_image[...,::-1])
                    angles.append(center_angle)
                    images.append(image_flipped[...,::-1])
                    angles.append(measurement_flipped)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

if __name__ == '__main__':
    # get samples from csv file
    samples = []
    with open('./driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    
    print("Found Samples: ", len(samples))

    # split samples into training and validation sets
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
 
    # build model
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Crop top and bottom off images
    model.add(Cropping2D(cropping=((70,50), (0,0))))

    # convolutional layers
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT_RATE))

    # dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1)) # regression not classification, no activation

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 6, validation_data=validation_generator, nb_val_samples=len(validation_samples) * 6, nb_epoch=1)

    print(history_object.history.keys())
    print(history_object.history['loss'])
    print(history_object.history['val_loss'])
    model.save("model.h5")