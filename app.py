from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

interpreter = tf.lite.Interpreter(model_path="flower_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
img_size = (224, 224)

class_names = ['Acteria', 'Blue Water Lilly', 'Hibiscus', 'Rukkaththana', 'unknown']
flower_info = {
    "Acteria": ("ඇක්ටේරියා", "Murraya paniculata"),
    "Blue Water Lilly": ("නිල් මානෙල්", "Nymphaea nouchali"),
    "Hibiscus": ("පොකුරු වද", "Hibiscus rosa-sinensis"),
    "Rukkaththana": ("රුක්කත්තන", "Allamanda cathartica"),
    "unknown": ("නොහඳුනන ලදි", "Not recognized")
}

def predict(image_path, threshold=0.7):
    try:
        with Image.open(image_path).convert("RGB") as img:
            img = img.resize(img_size)
            img = np.array(img) / 127.5 - 1.0
            img = np.expand_dims(img.astype(np.float32), axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        confidence = float(np.max(prediction))
        class_index = int(np.argmax(prediction))

        if confidence < threshold:
            return {"sinhala": "නොහඳුනන ලදි", "scientific": "Not recognized", "confidence": confidence}

        label = class_names[class_index]
        sinhala, scientific = flower_info[label]
        return {"sinhala": sinhala, "scientific": scientific, "confidence": confidence}

    except (UnidentifiedImageError, OSError):
        return {"sinhala": "නොහඳුනන ලදි", "scientific": "Invalid image", "confidence": 0.0}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict_flower():
    if 'image' not in request.files:
        return render_template("index.html", result={"sinhala": "නොහඳුනන ලදි", "scientific": "No image uploaded", "confidence": 0.0})

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict(filepath)

    # Optionally delete after prediction
    try:
        os.remove(filepath)
    except Exception as e:
        print(f"⚠️ Could not delete file: {e}")

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
