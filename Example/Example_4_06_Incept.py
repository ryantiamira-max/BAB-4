
# Example 4.6 - InceptionV3

from tensorflow.keras.applications import inception_v3

# Initialize the model
model = inception_v3.InceptionV3(weights='imagenet')

# Display model architecture
model.summary()

