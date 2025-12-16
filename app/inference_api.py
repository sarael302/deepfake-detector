# app/inference_api.py

import uvicorn
from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
import cv2
import os
from typing import Tuple
app = FastAPI(title="Deepfake Detector API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tu peux mettre les domaines spécifiques si tu veux
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================
# Load model once at startup
# ============================

MODEL_PATH = "deepfake_xception.h5"

print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Xception input size
IMG_SIZE = (299, 299)


# ============================
# Utils
# ============================

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (299, 299))

    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.xception.preprocess_input(img)
    return img



# ============================
# API ROUTES
# ============================

@app.get("/")
def home():
    return {"message": "Deepfake Detector API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Send an image => returns fake/real + confidence
    """
    bytes_data = await file.read()
    img = preprocess_image(bytes_data)

    # Model prediction
    prob = float(model.predict(img)[0][0])
    label = "Real" if prob >= 0.5 else "Fake"

    return {
        "label": label,
        "confidence": round(prob if prob >= 0.5 else 1 - prob, 4)
    }


# ============================
# Run API (dev mode)
# ============================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
