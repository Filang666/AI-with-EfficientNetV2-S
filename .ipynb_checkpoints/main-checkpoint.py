import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="EfficientNetV2 Image Classifier API")

# --- 1. Configuration ---
IMG_SIZE = (300, 300)
# Load class names dynamically
try:
    with open('classes.txt', 'r') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    CLASS_NAMES = ["Unknown"] * 10
# Load model globally to avoid reloading on every request (crucial for performance)
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e:
    # If model is missing during local dev, we handle it to prevent crash
    model = None
    print(f"Warning: model.h5 not found or corrupted. Error: {e}")

# Pydantic schema for structured API responses (Standard for Middle level)
class PredictionResponse(BaseModel):
    class_name: str
    confidence: float

@app.get("/health")
def health_check():
    """Health probe for Docker/Kubernetes orchestrators."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Accepts an image and returns the predicted class with confidence score."""
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG/PNG allowed.")

    if model is None:
        raise HTTPException(status_code=503, detail="Inference model is not initialized.")

    try:
        # 2. Image Preprocessing
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize(IMG_SIZE)
        
        # Convert to numpy and add batch dimension (1, 300, 300, 3)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 

        # 3. Model Inference
        predictions = model.predict(img_array, verbose=0)
        
        idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[idx]
        confidence = float(predictions[0][idx])

        return PredictionResponse(
            class_name=predicted_class,
            confidence=round(confidence * 100, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
