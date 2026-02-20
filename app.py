import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model("pneumonia_model.keras")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", prediction="No file selected")

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    confidence = round(float(pred) * 100, 2)

    if pred > 0.5:
        result = "PNEUMONIA"
        confidence = confidence
    else:
        result = "NORMAL"
        confidence = 100 - confidence

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        image_path=file_path
    )

if __name__ == "__main__":
    app.run(debug=True)
