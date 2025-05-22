import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your model at startup so it persists in the function’s global scope.
MODEL_PATH = os.path.join(os.getcwd(), 'batik.h5')
model = load_model(MODEL_PATH)

CLASS_LABELS = ["bokor-kencono", "truntum"]

def preprocess_image(image_bytes):
    from io import BytesIO
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = img.size
    new_edge = min(width, height)
    left = (width - new_edge) // 2
    top = (height - new_edge) // 2
    right = left + new_edge
    bottom = top + new_edge
    img = img.crop((left, top, right, bottom))
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def handler(request):
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Only POST method is supported."})
        }
    try:
        # Get the uploaded image from the request body
        image_bytes = request.body  # Ensure this returns the file bytes as expected
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_LABELS[predicted_idx]
        confidence = float(predictions[0][predicted_idx]) * 100
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

    response = {
        "predicted_class": predicted_class,
        "confidence": confidence
    }
    return {
         "statusCode": 200,
         "body": json.dumps(response)
    }