# -*- coding: utf-8 -*-
"""
initial CNN model to classify images of flowers
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout

rate = 0.2

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape=(256, 256, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(256,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(rate, noise_shape=None, seed=None))

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(rate, noise_shape=None, seed=None))

model.add(Dense(512, activation = 'relu'))
model.add(Dropout(rate, noise_shape=None, seed=None))

model.add(Dense(7, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'flowers/training_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'flowers/test_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')


model.fit_generator(
        training_set,
        steps_per_epoch=5306/32,
        epochs=25,
        validation_data=test_set,
        validation_steps=216/32)

model.save('new.h5')

from keras.models import load_model
model = load_model('new.h5')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('flowers/single_prediction/download.jpg', target_size=(256, 256) )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
prediction = np.max(result)
if result[0][0] == prediction:
    print('The Image is that of a Jasmine flower')
elif result[0][1] == prediction:
    print('The Image is that of a Daisy flower')
elif result[0][2] == prediction:
    print('The Image is that of a Dandelion flower')
elif result[0][3] == prediction:
    print('The Image is that of a Lotus flower')
elif result[0][4] == prediction:
    print('The Image is that of a Rose flower')
elif result[0][5] == prediction:
    print('The Image is that of a Sunflower')
elif result[0][6] == prediction:
    print('The Image is that of a Tulip')
