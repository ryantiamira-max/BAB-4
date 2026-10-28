
# Example 4.16 AutoencoderKeras.py
# Modified from:
# https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/

# Load the MNIST handwritten digit images
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(x_train.shape)
print(x_test.shape)

# Build Autoencoder
from keras.layers import Dense, Flatten, Reshape, Input
from keras.models import Sequential, Model

def build_autoencoder(img_shape, code_size):
    # Encoder
    encoder = Sequential([
        Input(shape=img_shape),
        Flatten(),
        Dense(code_size)
    ])

    # Decoder
    decoder = Sequential([
        Input(shape=(code_size,)),
        Dense(int(np.prod(img_shape))),
        Reshape(img_shape)
    ])

    return encoder, decoder

IMG_SHAPE = x_train[0].shape
encoder, decoder = build_autoencoder(IMG_SHAPE, 32)

inp = Input(shape=IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

autoencoder.summary()

# Train Autoencoder
history = autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=20,
    validation_data=(x_test, x_test),
    batch_size=256
)

# Plot training results
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot original, encoded, and reconstructed images
def show_image(x):
    plt.imshow(np.clip(x, 0, 1), cmap='gray')
    plt.axis('off')

def visualize(img, encoder, decoder):
    # Encode and reconstruct image
    code = encoder.predict(img[None, ...], verbose=0)[0]
    reco = decoder.predict(code[None, ...], verbose=0)[0]

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1, 3, 2)
    plt.title("Code")
    plt.imshow(code.reshape(4, 8), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)

    plt.tight_layout()
    plt.show()

for i in range(5):
    img = x_test[i]
    visualize(img, encoder, decoder)
