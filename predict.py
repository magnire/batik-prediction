from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')

# Load model with error handling
MODEL_PATH = 'model/batik.h5'
try:
    logger.info(f"Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

@app.route('/')
def home():
    logger.info("Home route accessed")
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Predict route accessed")
    try:
        # Check if model is loaded
        if model is None:
            raise Exception("Model not loaded")

        # Verify request contains file
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({"error": "No image file provided"}), 400

        # Get the image file from the request
        file = request.files['image']
        logger.info(f"Received image file: {file.filename}")
        
        # Open and preprocess image
        image = Image.open(file.stream).convert('RGB')
        logger.info(f"Image opened, size: {image.size}")
        
        # Preprocess the image
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        logger.info(f"Preprocessed image shape: {img_array.shape}")
        
        # Make prediction
        logger.info("Making prediction...")
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        
        # Define class labels
        class_labels = ["bokor-kencono", "truntum"]
        predicted_class = class_labels[predicted_idx]
        confidence = float(predictions[0][predicted_idx]) * 100
        
        logger.info(f"Prediction complete: {predicted_class} ({confidence:.2f}%)")
        
        response = {
            "predicted_class": predicted_class,
            "confidence": confidence
        }
        logger.info(f"Returning response: {response}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error in predict route: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full exception: {sys.exc_info()}")
        return jsonify({"error": error_msg}), 500

# Add error handlers
@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(e):
    logger.error(f"Route not found: {request.url}")
    return jsonify({"error": "Route not found"}), 404

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)