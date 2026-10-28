# Exercise 4.4 - LeNet dengan jumlah filter berbeda (6->8, 16->32)
# Bandingkan Total params dengan Example 4.4 (81,194)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,AveragePooling2D
from keras import optimizers

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
input_shape=(32,32,1)))
model.add(AveragePooling2D())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation = 'softmax'))
model.summary()
