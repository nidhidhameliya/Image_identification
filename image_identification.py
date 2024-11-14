import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Set image dimensions and paths
img_height = 150
img_width = 150

train_data_dir = 'C:/Users/Nidhi Dhameliya/Desktop/jupyter/imagedata/training'
validation_data_dir = 'C:/Users/Nidhi Dhameliya/Desktop/jupyter/imagedata/validation'
batch_size = 32

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Just normalization for validation
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Training and validation generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 20
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the model in the native Keras format
model.save('laptop_mobile_classifier.keras')

# Load and predict on a new image
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(img_path, model):
    if os.path.isfile(img_path):
        try:
            img = image.load_img(img_path, target_size=(img_height, img_width))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            prediction = model.predict(img_array)
            if prediction[0] > 0.75:  # Adjust the threshold as needed
                print("It's a mobile!")
            elif prediction[0] < 0.25:  # Adjust the threshold as needed
                print("It's a laptop!")
            else:
                print("The image is neither a mobile nor a laptop!")
        except Exception as e:
            print(f"An error occurred while processing the image: {e}")
    else:
        print(f"File not found: {img_path}")

# Example usage
image_path = 'C:/Users/Nidhi Dhameliya/Desktop/jupyter/imagedata/testing/1.png'
predict_image(image_path, model)