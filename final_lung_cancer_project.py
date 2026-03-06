# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')

# Data manipulation and visualization
import pandas as pd
print(pd.__version__)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning preprocessing and algorithms
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten, Conv2D,
                                      MaxPooling2D, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Paths to dataset folders
train_folder = r'dataset/train'
validate_folder = r'dataset/valid'

# Check if dataset paths exist
if not os.path.exists(train_folder) or not os.path.exists(validate_folder):
    raise FileNotFoundError("Check the dataset paths. Make sure they exist!")

print("Training folder content:", os.listdir(train_folder))
print("Validation folder content:", os.listdir(validate_folder))

# Image data generators
IMAGE_SIZE = (350, 350)
batch_size = 8

# Data augmentation for training
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

# Only rescaling for validation
test_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='categorical'
)

# Validation data generator
validation_generator = test_datagen.flow_from_directory(
    validate_folder,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the number of output classes
OUTPUT_SIZE = train_generator.num_classes

# Load a pre-trained model (e.g., Xception)
pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
pretrained_model.trainable = False  # Freeze base model

# Build the complete model
model = Sequential([
    pretrained_model,
    GlobalAveragePooling2D(),
    Dense(OUTPUT_SIZE, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model created successfully!")
model.summary()

# Callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.5, min_lr=1e-6, verbose=2)
early_stops = EarlyStopping(monitor='loss', patience=6, verbose=2)
checkpointer = ModelCheckpoint(filepath='best_model.h5', save_best_only=True, verbose=2)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=20,
    callbacks=[learning_rate_reduction, early_stops, checkpointer]
)

# Save the model
model.save('lung_cancer_detection_model.h5')
print("Model saved successfully!")

# Plot training curves
def display_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:
        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('Model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Valid'])

plt.figure(figsize=(10, 5))
display_training_curves(history.history['loss'], history.history['val_loss'], 'Loss', 211)
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy', 212)
plt.show()

# Prediction function
from tensorflow.keras.preprocessing import image

def predict_image(model, img_path, target_size, class_labels):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return class_labels[predicted_class]

# # Predict a sample image
# class_labels = list(train_generator.class_indices.keys())

# img_path = r'path\to\test_image.jpg'  # Change this to the path of your test image
# if os.path.exists(img_path):
#     predicted_label = predict_image(model, img_path, IMAGE_SIZE, class_labels)
#     print(f"The predicted class is: {predicted_label}")
# else:
#     print(f"Test image not found at {img_path}")
