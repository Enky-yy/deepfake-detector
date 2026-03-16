import cv2 as cv

def extract_frames(video_path, max_frames=20):
    cap = cv.VideoCapture(video_path)
    frames =[]
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    steps = max(total // max_frames , 1)
    count =0

    while True:
        ret , frame =cap.read()
        if not ret:
            break

        if count % steps == 0:
            frames.append(frame)
            count+=1

        if len(frame)>= max_frames:
            break

        cap.release()

    return frames