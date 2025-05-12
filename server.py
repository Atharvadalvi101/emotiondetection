from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read raw image data from ESP32-CAM
        image_data = request.data
        np_arr = np.frombuffer(image_data, np.uint8)

        # Decode the raw byte image to grayscale
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Check if the image is of the correct size, resize if necessary
        if img.shape != (48, 48):
            img = cv2.resize(img, (48, 48))
        
        # Normalize and reshape image for inference
        img = img.astype(np.float32) / 255.0
        img = img.reshape(1, 48, 48, 1)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        prediction = labels[np.argmax(output)]

        return jsonify({"emotion": prediction})
    
    except Exception as e:
        print(f"Error: {e}")  # Logs detailed error on the server side
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Emotion Detection Server (Grayscale TFLite) is Running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
