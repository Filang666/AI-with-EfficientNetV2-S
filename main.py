import io

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from config import IMG_SIZE, MODEL_PATH, get_class_names

app = FastAPI(title="AI Image Classifier")

# Load assets once
CLASS_NAMES = get_class_names()
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


class Prediction(BaseModel):
    label: str
    confidence: float


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type")

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB").resize(IMG_SIZE)
        img_array = np.expand_dims(
            tf.keras.preprocessing.image.img_to_array(img), axis=0
        )

        preds = model.predict(img_array, verbose=0)[0]
        idx = np.argmax(preds)

        return Prediction(
            label=CLASS_NAMES[idx], confidence=round(float(preds[idx]) * 100, 2)
        )
    except Exception as e:
        raise HTTPException(500, f"Processing error: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}
