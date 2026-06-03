from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
from src.utils.download_models import ensure_models
from src.inference.predictor import predict_multimodal

app = FastAPI()
@app.on_event("startup")
async def startup_event():
    ensure_models()
    
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Multimodal Emotion Recognition API Running"
    }


@app.post("/predict-emotion")
async def predict_emotion(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    text: str = Form(...)
):

    # save image
    image_path = os.path.join(
        TEMP_DIR,
        image.filename
    )

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(
            image.file,
            buffer
        )

    # save audio
    audio_path = os.path.join(
        TEMP_DIR,
        audio.filename
    )

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(
            audio.file,
            buffer
        )

    result = predict_multimodal(
        image_path=image_path,
        audio_path=audio_path,
        text_input=text
    )

    return result
