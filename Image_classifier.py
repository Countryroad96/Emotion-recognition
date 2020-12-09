#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np   
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

model_dir = "/model/modelname.h5"
img_dir = "/pictures/filename.jpg"


# Face detection XML load and trained model loading
face_detection = cv2.CascadeClassifier('/haarcascade_file/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(model_dir, compile=False)
EMOTIONS = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Image path
img = cv2.imread(img_dir)

# Convert color to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Face detection in frame
faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

# Create empty image
canvas = np.zeros((250, 300, 3), dtype="uint8")

# Perform emotion recognition only when face is detected
if len(faces) > 0:
    # For the largest image
    face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = face
    # Resize the image to 48x48 for neural network
    roi = gray[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # Emotion predict
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]

    # Assign labeling
    cv2.putText(img, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.rectangle(img, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    # Label printing
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        prob = prob.astype(np.float32)        
        text = "{}: {}%".format(emotion, np.floor(prob*10000)/100)
        w = prob * 300
        w = w.astype(np.int32)
        cv2.rectangle(canvas, (6, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Open two windows
## Display image ("Emotion Recognition")
## Display probabilities of emotion
cv2.imshow('Emotion Recognition', img)
cv2.imshow("Probabilities", canvas)

# q to quit
cv2.waitKey()

# Clear program and close windows
cv2.destroyAllWindows()

