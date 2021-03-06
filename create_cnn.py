# Code mostly taken from
# https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
# Dataset taken from https://www.kaggle.com/tongpython/cat-and-dog

# Importing the Keras libraries and packages
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from tensorflow import keras

# Initialising the CNN
model = Sequential()
# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
# Step 2 - Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a second convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Step 3 - Flattening
model.add(Flatten())
# Step 4 - Full connection
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# Compiling the CNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 32

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(128, 128),
                                                 batch_size=batch_size,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(128, 128),
                                            batch_size=batch_size,
                                            class_mode='binary')

import os

cp_callback = keras.callbacks.ModelCheckpoint(filepath=os.curdir,
                                              save_weights_only=True,
                                              verbose=1)

model.fit(training_set,
          steps_per_epoch=8000 // batch_size,
          epochs=25,
          validation_data=test_set,
          validation_steps=2000 // batch_size,
          callbacks=[cp_callback])

model.save("keras_cnn")
print("Model saved")
