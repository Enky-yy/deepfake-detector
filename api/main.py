from fastapi import FastAPI, UploadFile
from inference.video_predictor import predict_video

app =FastAPI()

@app.post('/detect')

async def detect(file:UploadFile):

    result = predict_video(file.file)

    return {'DeepFake_probability' : float(result)}