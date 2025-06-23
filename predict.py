from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load model
model = load_model("healthy_vs_rotten.h5")

# Class labels
class_labels = ['Banana__Healthy', 'Banana__Rotten', 'Bellpepper__Healthy', 'Bellpepper__Rotten',
                'Carrot__Healthy', 'Carrot__Rotten', 'Cucumber__Healthy', 'Cucumber__Rotten',
                'Grape__Healthy', 'Grape__Rotten', 'Guava__Healthy', 'Guava__Rotten',
                'Jujube__Healthy', 'Jujube__Rotten', 'Mango__Healthy', 'Mango__Rotten',
                'Orange__Healthy', 'Orange__Rotten', 'Pomegranate__Healthy', 'Pomegranate__Rotten',
                'Potato__Healthy', 'Potato__Rotten', 'Strawberry__Healthy', 'Strawberry__Rotten',
                'Tomato__Healthy', 'Tomato__Rotten']

# ✅ Use exact image filename (use r'' to avoid escape warnings)
img_path = r"E:\ANUSHA\VANGMAYEE\smart-sorting\output_dataset\test\Banana__Healthy\rotated_by_45_Screen Shot 2018-06-12 at 9.39.53 PM.png"

# Load and preprocess
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = preprocess_input(x)
x = np.expand_dims(x, axis=0)

# Predict
pred = model.predict(x)
index = np.argmax(pred)

print("Predicted class index:", index)
print("Predicted label:", class_labels[index])
