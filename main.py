import io
import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from config import IMG_SIZE, MODEL_PATH, get_class_names
from model_factor import predict_with_ood

app = FastAPI(title="AI Image Classifier")

# Load assets
CLASS_NAMES = get_class_names()
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        # Pre-warmup
        model(tf.zeros((1, *IMG_SIZE, 3)))
        print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    entropy: float
    is_ood: bool


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predicts image class and detects Out-of-Distribution data."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB").resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Unified OOD logic
        idx, conf, entropy, is_ood = predict_with_ood(model, img_tensor)

        if is_ood:
            label = "Unknown / Out-of-Distribution"
        else:
            label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"

        return PredictionResponse(
            label=label, confidence=conf, entropy=round(entropy, 4), is_ood=is_ood
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_ready": model is not None}
