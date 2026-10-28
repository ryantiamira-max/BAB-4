
# Example 4.9
# Simple CNN for the MNIST Dataset

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (
    Input, Dense, Dropout, Flatten,
    Conv2D, MaxPooling2D
)
from keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Plot 4 images in grayscale
plt.figure(figsize=(10, 3))

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.axis('off')

plt.show()

# Reshape images to [samples, height, width, channels]
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

# Normalize pixel values from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode outputs
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
num_classes = y_test.shape[1]

# Create model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200
)

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
