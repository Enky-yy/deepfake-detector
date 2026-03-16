import os 
import cv2 as cv
from face_extraction import extract_frames
from face_detection import detect_face

def preprocess(input_dir, output_dir):
    os.makedirs(output_dir,exist_ok=True)

    count = 0 

    for video in os.listdir(input_dir):

        video_path = os.path.join(input_dir,video)

        print('preprocessing:', video)

        frames = extract_frames(video_path=video_path)

        for frame in frames:

            face = detect_face(frame=frame)

            if face is None:
                continue

            face = cv.resize(face ,(224,224))

            save_path = os.path.join(output_dir, f"{count}.jpg")

            cv.imwrite(save_path,face)

            count +=1

def main():
    preprocess(, "df/real")
    
    preprocess(, "df/fake")

if __name__ == "__main__":
    main()