import tensorflow as tf
from datasets.dataset_loader import process_video
from keras import models

model = models.load_model('trained_models/model.h5')

def predict_video(path):
        
    faces = process_video(path)

    faces = faces[None,...]

    prediction = model.predict(faces)

    return prediction.mean()