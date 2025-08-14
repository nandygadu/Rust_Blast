from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Define a flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  # Choose the folder where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the Keras model
MODEL_PATH = 'Blast_Rust_Model.keras'
loaded_model = load_model(MODEL_PATH)
loaded_model.make_predict_function()  # Necessary

# Image preprocessing function
def preprocess_image(image_path, image_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Prediction function
def predict_image(img_array, cnn_model):
    prediction = cnn_model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    probability = prediction[0][predicted_class_index]
    return predicted_class_index, probability

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Save the uploaded image
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image
            image_size = (224, 224)
            img_array = preprocess_image(file_path, image_size)

            # Predict the image
            predicted_class_index, probability = predict_image(img_array, loaded_model)

            return render_template('result.html', filename=filename, predicted_class_index=predicted_class_index, probability=probability)



if __name__ == '__main__':
    app.run(debug=True)