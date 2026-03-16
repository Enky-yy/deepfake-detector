import cv2 as cv
import mediapipe as mp

mp_face = mp.solution.face_detection
dectector = mp_face.FaceDetection()

def detect_face(frame):
    rgb_color = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    results = dectector.process(rgb_color)

    if not results.detections:
        return None
    
    bbox = results.detection[0].location_data.relative_bounding_box

    h,w = frame.shape

    x=int(bbox.xmin*w)
    y= int (bbox.ymin*h)
    w = int (bbox.width*w)
    h = int (bbox.height*h)

    face = frame[y:y+h, x:x+w]

    return face