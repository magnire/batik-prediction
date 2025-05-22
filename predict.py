from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__, static_folder='.', static_url_path='')

# Load model once when the server starts
MODEL_PATH = 'model/batik.h5'
model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        # Preprocess the image
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        
        # Define class labels
        class_labels = ["bokor-kencono", "truntum"]
        predicted_class = class_labels[predicted_idx]
        confidence = float(predictions[0][predicted_idx]) * 100
        
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)