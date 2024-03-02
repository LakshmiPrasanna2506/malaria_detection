
from flask import Flask, render_template, request
import os
from keras.preprocessing import image
from keras.models import load_model
from werkzeug.utils import secure_filename  # Import secure_filename
import numpy as np
from PIL import Image
app = Flask(__name__)

model = load_model('C:/Users/manis/Downloads/malaria-detection-main/malaria-detection-main/vgg.h5')

def predict_class(img):
    x = image.img_to_array(img)
    x = x/255.0
    x = np.expand_dims(x,axis=0)
    proba = model.predict(x)[0][0]

    y = "Uninfected" if proba > 0.5 else "Parasitized"

    if y == "Parasitized":
        proba = 1 - proba

    return y, round(proba * 100, 3)

UPLOAD_FOLDER = "C:/Users/manis/OneDrive/Desktop/UPLOAD_FOLDER"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=['GET', 'POST'])
def upload_predict():
    if request.method == "POST":
        image_file = request.files['image']
        if image_file:
            # Save the uploaded image to the UPLOAD_FOLDER
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
            image_file.save(img_path)

            # Load and preprocess the uploaded image
            img = image.load_img(img_path, target_size=(150, 150))
            x = image.img_to_array(img)
            x = x / 255.0
            x = np.expand_dims(x, axis=0)

            # Make a prediction
            proba = model.predict(x)[0][0]
            y = "Uninfected" if proba > 0.5 else "Parasitized"

            if y == "Parasitized":
                proba = 1 - proba

            return render_template("home.html", prediction=y, img_loc=img_path, proba=round(proba * 100, 3))

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True, port=8000)