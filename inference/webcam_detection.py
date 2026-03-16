import cv2 as cv
from preprocessing.face_detection import detect_face
import tensorflow as tf
from keras import models
from models import load_models as load

model = load('trained_models/model.h5')

cap = cv.VideoCapture(0)

while True:

    ret , frame = cap.read()

    face = detect_face(frame)

    if face is not None:
        face = cv.resize(face,(224,224))/255.0

        preds = model.predict(face)

        label = "Fake"  if preds>=0.55 else 'Real'

        cv.putText(frame, label, (20,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv.imshow('DeepFake Detector', frame)

    if cv.waitKey(1)==27:
        break