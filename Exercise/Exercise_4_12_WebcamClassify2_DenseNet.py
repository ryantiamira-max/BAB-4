
# Exercise 4.12 - Tambah DenseNet ke Example 4.14

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3,
    densenet
)

# Initialize model
# Uncomment one model if you want to compare architectures
# model = vgg16.VGG16(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')
# model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')

model = densenet.DenseNet121(weights='imagenet')
model.summary()

# Open webcam
camera = cv2.VideoCapture(0)
image_size = 224

if not camera.isOpened():
    print("Webcam tidak dapat dibuka.")
    exit()

try:
    while True:
        ok, cam_frame = camera.read()

        if not ok:
            print("Gagal membaca frame dari webcam.")
            break

        # Resize image
        frame = cv2.resize(cam_frame, (image_size, image_size))
        numpy_image = img_to_array(frame)
        image_batch = np.expand_dims(numpy_image, axis=0)

        # Preprocess image for DenseNet121
        processed_image = densenet.preprocess_input(image_batch.copy())

        # Predict class
        predictions = model.predict(processed_image, verbose=0)
        label = decode_predictions(predictions, top=1)[0][0]

        class_name = label[1]
        confidence = label[2] * 100

        # Display prediction
        text = f"{class_name}, {confidence:.1f}%"
        cv2.putText(
            cam_frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 0, 0), 2
        )

        cv2.imshow('DenseNet121 Webcam Classification', cam_frame)

        # Press ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
