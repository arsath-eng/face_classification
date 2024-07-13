import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Function to load model from Hugging Face Hub
def load_model_from_hub(repo_id, filename):
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return tf.keras.models.load_model(model_path)

# Load models from Hugging Face Hub
model1 = load_model_from_hub("arsath-sm/face_classification_model1", "face_classification_model1.h5")
model2 = load_model_from_hub("arsath-sm/face_classification_model2", "face_classification_model2.h5")

def preprocess_image(image):
    img = tf.image.resize(image, (224, 224))  # Resize to match the input size of your models
    img = tf.cast(img, tf.float32) / 255.0  # Normalize pixel values
    return tf.expand_dims(img, 0)  # Add batch dimension

def predict_image(image):
    preprocessed_image = preprocess_image(image)
    # Make predictions using both models
    pred1 = model1.predict(preprocessed_image)[0][0]
    pred2 = model2.predict(preprocessed_image)[0][0]
    
    # Prepare results for each model
    result1 = "Real" if pred1 > 0.5 else "Fake"
    confidence1 = pred1 if pred1 > 0.5 else 1 - pred1
    result2 = "Real" if pred2 > 0.5 else "Fake"
    confidence2 = pred2 if pred2 > 0.5 else 1 - pred2
    
    return {
        "model1": {"result": result1, "confidence": float(confidence1)},
        "model2": {"result": result2, "confidence": float(confidence2)}
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            image = tf.image.decode_image(file.read())
            results = predict_image(image)
            return jsonify(results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)