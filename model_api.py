from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import shutil
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# -----------------------
# MODEL YOLU VE İNDİRME
# -----------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "beyin_tumoru_modeli.h5")
model_url = "https://huggingface.co/Mortines/BrainAIModel/resolve/main/beyin_tumoru_modeli.h5"

# Modeli indir ve kaydet (sadece bir kez)
if not os.path.exists(model_path):
    print("📥 Hugging Face'ten model indiriliyor...")
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Model indirildi.")
    except Exception as e:
        raise RuntimeError(f"Model indirilemedi: {str(e)}")

# Modeli yükle
model = load_model(model_path)

# -----------------------
# KATEGORİLER
# -----------------------

categories = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'no_tumor']

# -----------------------
# DTO (Gelen veri yapısı)
# -----------------------

class PostRequest(BaseModel):
    url: str

# -----------------------
# TAHMİN FONKSİYONU
# -----------------------

def predict_tumor(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index]) * 100

    turkce_karsilik = {
        'glioma_tumor': 'Glioma tümörü tespit edildi.',
        'meningioma_tumor': 'Meningioma tümörü tespit edildi.',
        'pituitary_tumor': 'Hipofiz tümörü tespit edildi.',
        'no_tumor': 'Tümör tespit edilmedi.'
    }

    predicted_class = categories[class_index]
    return turkce_karsilik[predicted_class], round(confidence, 2)

# -----------------------
# API UÇ NOKTASI
# -----------------------

@app.post("/predict")
async def predict(post_request: PostRequest):
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        response = requests.get(post_request.url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Resim indirilemedi.")

        image_path = os.path.join(temp_dir, "temp_image.jpg")
        with open(image_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)

        label, confidence = predict_tumor(image_path)
        os.remove(image_path)

        return {
            "prediction": label,
            "confidence": f"{confidence:.2f}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
