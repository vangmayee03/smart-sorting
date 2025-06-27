from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tensorflow as tf
import uuid

app = Flask(__name__)

model = tf.keras.models.load_model('healthvsrotten.h5')

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class index list
index = [
    'Banana__Healthy', 'Banana__Rotten', 'Bellpepper__Healthy', 'Bellpepper__Rotten',
    'Carrot__Healthy', 'Carrot__Rotten', 'Cucumber__Healthy', 'Cucumber__Rotten',
    'Grape__Healthy', 'Grape__Rotten', 'Guava__Healthy', 'Guava__Rotten',
    'Jujube__Healthy', 'Jujube__Rotten', 'Mango__Healthy', 'Mango__Rotten',
    'Orange__Healthy', 'Orange__Rotten', 'Pomegranate__Healthy', 'Pomegranate__Rotten',
    'Potato__Healthy', 'Potato__Rotten', 'Strawberry__Healthy', 'Strawberry__Rotten',
    'Tomato__Healthy', 'Tomato__Rotten'
]

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Grab form data
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Print to console (or save/send)
        print("New Contact Message:")
        print("Name:", name)
        print("Email:", email)
        print("Message:", message)

        return render_template('contact.html', success=True)

    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['pc_image']
        if not file:
            return redirect(url_for('predict'))

        # Save with unique filename
        filename = str(uuid.uuid4()) + "_" + file.filename
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        # Preprocess
        image = load_img(path, target_size=(224, 224))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        pred = np.argmax(model.predict(image_array), axis=1)[0]
        label = index[pred]

        result = "HEALTHY" if "Healthy" in label else "ROTTEN"

        return render_template('portfolio-details.html', result=result, image_url=path)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
