# Exercise 4.14 - Example 4.15 dengan if-else untuk memilih model
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from classification_models.keras import Classifiers

import numpy as np
import cv2

#Pilih model: 'vgg16', 'resnet50', 'mobilenetv2', 'densenet201', 'inceptionv3'
model_name = 'vgg16'

if model_name == 'vgg16':
    clf, preprocess_input = Classifiers.get('vgg16')
    sz = 224
elif model_name == 'resnet50':
    clf, preprocess_input = Classifiers.get('resnet50')
    sz = 224
elif model_name == 'mobilenetv2':
    clf, preprocess_input = Classifiers.get('mobilenetv2')
    sz = 224
elif model_name == 'densenet201':
    clf, preprocess_input = Classifiers.get('densenet201')
    sz = 224
else:
    clf, preprocess_input = Classifiers.get('inceptionv3')
    sz = 299
image_size = sz
model = clf(input_shape=(sz,sz,3), weights='imagenet', classes=1000)

model.summary()
camera = cv2.VideoCapture(0)
while True:
    ret, cam_frame = camera.read()
    frame= cv2.resize(cam_frame, (image_size, image_size))
    image = np.asarray(frame)
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)

    preds = model.predict(image)
    label = decode_predictions(preds)

    cv2.putText(cam_frame, "{}, {:.1f}".format(label[0][0][1], label[0][0][2]),
        (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", cam_frame)
    key = cv2.waitKey(30)
    if key == 27: # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
