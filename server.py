from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def preprocess_image(image):
    image = image.convert("L").resize((48, 48))  # Grayscale + Resize
    img = np.array(image).astype(np.float32) / 255.0
    img = img.reshape(1, 48, 48, 1)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    input_data = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = labels[np.argmax(output)]

    return jsonify({"emotion": prediction})

@app.route("/", methods=["GET"])
def home():
    return "Emotion Detection Server is Running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
