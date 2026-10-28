
# Exercise 4.11 - Webcam classification dengan VGG19

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions

image_size = 224

# Load pretrained VGG19 model
model = vgg19.VGG19(weights='imagenet')
model.summary()

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam tidak dapat dibuka. Periksa kamera atau indeks kamera.")
    exit()

try:
    while True:
        ok, cam_frame = camera.read()

        if not ok:
            print("Gagal membaca gambar dari webcam.")
            break

        # Resize frame for VGG19
        frame = cv2.resize(cam_frame, (image_size, image_size))

        # Convert image to array and add batch dimension
        numpy_image = img_to_array(frame)
        image_batch = np.expand_dims(numpy_image, axis=0)

        # Preprocess image
        processed_image = vgg19.preprocess_input(image_batch.copy())

        # Predict object class
        predictions = model.predict(processed_image, verbose=0)
        label = decode_predictions(predictions, top=1)[0][0]

        class_name = label[1]
        confidence = label[2] * 100

        # Display prediction
        text = f"VGG19: {class_name}, {confidence:.1f}%"
        cv2.putText(
            cam_frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 0, 0), 2
        )

        cv2.imshow('VGG19 Webcam Classification', cam_frame)

        # Press ESC to quit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
