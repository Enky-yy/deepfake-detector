import numpy as np 
import  cv2 as cv 
from preprocessing.face_detection import detect_face
from preprocessing.face_extraction import extract_frames

def process_video(path):

    frames = extract_frames(path)

    faces = []

    for frame in frames:

        face = detect_face(frame)

        if face is None:
            continue

        face = cv.resize(face,(224,224))

        faces.append(face)

    faces = np.array(faces)/255.0

    return faces