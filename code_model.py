#cell_1
import os
import shutil
from sklearn.model_selection import train_test_split

# Original dataset path
dataset_dir = r"C:\Users\prasa\OneDrive\Desktop\Fruit And Vegetable Diseases Dataset"
output_dir = r"C:\Users\prasa\OneDrive\Desktop\healthy_vs_rotten_dataset"

# Create output folder structure
for split in ['train', 'val', 'test']:
    for label in ['Healthy', 'Rotten']:
        os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

# Go through each class folder
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)

    if "__Healthy" in folder:
        label = "Healthy"
    elif "__Rotten" in folder:
        label = "Rotten"
    else:
        continue

    images = os.listdir(folder_path)[:200]  # limit to 200 images/class
    train_val, test = train_test_split(images, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)

    for img in train:
        shutil.copy(os.path.join(folder_path, img), os.path.join(output_dir, 'train', label, img))
    for img in val:
        shutil.copy(os.path.join(folder_path, img), os.path.join(output_dir, 'val', label, img))
    for img in test:
        shutil.copy(os.path.join(folder_path, img), os.path.join(output_dir, 'test', label, img))

print("✅ Dataset prepared at:", output_dir)
#cell_2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set paths
base_path = r"C:\Users\prasa\OneDrive\Desktop\healthy_vs_rotten_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(base_path, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(base_path, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load VGG16 base
vgg = VGG16(include_top=False, input_shape=(224, 224, 3))
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(128, activation='relu')(x)
output = Dense(2, activation='softmax')(x)  # 2 classes: Healthy, Rotten

model = Model(inputs=vgg.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stop]
)
#cell_3
model.save(r"C:\Users\prasa\OneDrive\Desktop\healthy_vs_rotten.h5")
print("✅ Model saved successfully.")
#cell_4
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load model
model = load_model(r"C:\Users\prasa\OneDrive\Desktop\healthy_vs_rotten.h5")

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    os.path.join(base_path, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Predictions
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

# Classification report
print(classification_report(y_true, y_pred_classes, target_names=labels))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
#cell_5
!pip install ipywidgets

#cell_6
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import tkinter as tk

# Load model
model = load_model(r"C:\Users\prasa\OneDrive\Desktop\healthy_vs_rotten.h5")
labels = ['Healthy', 'Rotten']

# Select image
tk.Tk().withdraw()
img_path = askopenfilename(title="Select an image", filetypes=[("Images", "*.jpg *.jpeg *.png")])

# Predict and display
if img_path:
    img = load_img(img_path, target_size=(224, 224))
    x = np.expand_dims(img_to_array(img), axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    label = labels[np.argmax(pred)]
    conf = round(np.max(pred) * 100, 2)

    print(f"✅ Prediction: {label} ({conf}%)")
    plt.imshow(img)
    plt.title(f"{label} ({conf}%)")
    plt.axis('off')
    plt.show()
else:
    print("❌ No image selected.")

