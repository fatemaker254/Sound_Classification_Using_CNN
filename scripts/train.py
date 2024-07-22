import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model
import numpy as np

# Directory containing the spectrogram images
data_dir = r"E:/SOUNDCLASSIFICATIONCNN/dataset"

# ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,  # 20% for validation
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training",
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="validation",
)

# Print class indices to verify labels
print("Class indices:", train_generator.class_indices)

# Print some batch data to verify correct loading
x_batch, y_batch = next(train_generator)
print("First batch images shape:", x_batch.shape)
print("First batch labels:", y_batch)

# Create the CNN model
model = create_model()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
)

# Save the trained model using joblib
joblib.dump(model, "elephant_sound_classifier.joblib")
