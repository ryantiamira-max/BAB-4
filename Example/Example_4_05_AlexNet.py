
# Example 4.5 AlexNet with Keras

import keras
from keras.models import Sequential
from keras.layers import (
    Input, Dense, Dropout, Flatten, Conv2D,
    MaxPooling2D, BatchNormalization
)

# Create the AlexNet model
model = Sequential()

model.add(Input(shape=(224, 224, 3)))

# 1st Convolutional Layer
model.add(Conv2D(
    filters=96, kernel_size=(11, 11),
    activation='relu', strides=(4, 4), padding='valid'
))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(
    filters=256, kernel_size=(5, 5),
    activation='relu', strides=(1, 1), padding='same'
))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(
    filters=384, kernel_size=(3, 3),
    activation='relu', strides=(1, 1), padding='same'
))

# 4th Convolutional Layer
model.add(Conv2D(
    filters=384, kernel_size=(3, 3),
    activation='relu', strides=(1, 1), padding='same'
))

# 5th Convolutional Layer
model.add(Conv2D(
    filters=256, kernel_size=(3, 3),
    activation='relu', strides=(1, 1), padding='same'
))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.4))

# Output Layer: 17 classes
model.add(Dense(17, activation='softmax'))

# Display model architecture
model.summary()

# Compile the model
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']
)

# Training and evaluation can be added when datasets are available
# model.fit(...)
# model.evaluate(...)
