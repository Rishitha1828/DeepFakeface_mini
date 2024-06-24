from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your deep fake detection model
model = tf.keras.models.load_model('C:/minipro/proj/cnn_deepfake_model.h5')

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)
    return image

def predict(image_path):
    try:
        image = Image.open(image_path)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        return prediction[0][0]  # Assuming binary classification (real/fake)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part in the request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No file selected")
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to {filepath}")
            prediction = predict(filepath)
            if prediction is not None:
                result = 'Fake' if prediction > 0.5 else 'Real'
                return render_template('result.html', result=result, filename=filename)
            else:
                print("Prediction error")
                return redirect(request.url)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)
