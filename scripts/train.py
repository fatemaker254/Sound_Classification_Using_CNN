import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

data_dir = r"E:/SOUNDCLASSIFICATIONCNN/dataset"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="training",
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="validation",
)

model = create_model()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
)

joblib.dump(model, "elephant_sound_classifier.joblib")
