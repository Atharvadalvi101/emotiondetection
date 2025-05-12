from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load the trained Keras .h5 model
model = tf.keras.models.load_model("best_model.h5")

# Emotion labels (update if your model uses different ones)
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def preprocess_image(image_bytes):
    # Read image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
    image = image.resize((48, 48))                            # Resize to 48x48
    img_array = np.array(image).reshape(1, 48, 48, 1)         # Shape to (1,48,48,1)
    img_array = img_array / 255.0                             # Normalize
    return img_array

@app.route("/", methods=["GET"])
def home():
    return "Emotion detection server is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        img = preprocess_image(image_bytes)
        prediction = model.predict(img)
        predicted_label = labels[np.argmax(prediction)]
        return jsonify({"emotion": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
