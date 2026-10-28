
# Example 4.15
# Image Classification using VGG16 and Webcam

from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers

import numpy as np
import cv2

# Set up model
clf, preprocess_input = Classifiers.get('vgg16')

sz = 224

model = clf(
    input_shape=(sz, sz, 3),
    weights='imagenet',
    classes=1000
)

model.summary()

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam tidak dapat dibuka.")
    exit()

try:
    while True:
        ret, cam_frame = camera.read()

        if not ret:
            print("Gagal membaca gambar dari webcam.")
            break

        # Resize frame
        frame = cv2.resize(cam_frame, (sz, sz))

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare image for prediction
        image = np.asarray(frame, dtype=np.float32)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Predict object
        preds = model.predict(image, verbose=0)
        label = decode_predictions(preds, top=1)

        # Display prediction
        class_name = label[0][0][1]
        confidence = label[0][0][2]

        cv2.putText(
            cam_frame,
            "{}, {:.1f}%".format(class_name, confidence * 100),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Classification", cam_frame)

        # Press ESC to quit
        key = cv2.waitKey(30)
        if key == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
