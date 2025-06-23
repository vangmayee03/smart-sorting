# -----------------------#
# 1. IMPORTS
# -----------------------#
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------#
# 2. DATA SPLITTING
# -----------------------#
dataset_dir = "E:/ANUSHA/VANGMAYEE/smart-sorting/Fruit And Vegetable Diseases Dataset"
output_dir = "output_dataset"

classes = os.listdir(dataset_dir)

os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

for cls in classes:
    os.makedirs(os.path.join(output_dir, 'train', cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', cls), exist_ok=True)

    class_dir = os.path.join(dataset_dir, cls)
    images = os.listdir(class_dir)[:200]

    train_val, test = train_test_split(images, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)

    for img in train:
        shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'train', cls, img))
    for img in val:
        shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'val', cls, img))
    for img in test:
        shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'test', cls, img))

print("✅ Dataset split complete.")

# -----------------------#
# 3. DATA GENERATORS
# -----------------------#
IMG_SIZE = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(output_dir, 'train'),
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical'  # ✅ use categorical for >2 classes
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(output_dir, 'val'),
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(output_dir, 'test'),
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# -----------------------#
# 4. MODEL BUILDING (VGG16)
# -----------------------#
vgg = VGG16(include_top=False, input_shape=(224, 224, 3))
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(128, activation='relu')(x)
output = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------#
# 5. TRAINING
# -----------------------#
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    steps_per_epoch=20,
    callbacks=[early_stop]
)

model.save("healthy_vs_rotten.h5")
print("✅ Model saved as healthy_vs_rotten.h5")

# -----------------------#
# 6. CLASS INDEX MAPPING
# -----------------------#
print("Class labels mapping:", train_generator.class_indices)
