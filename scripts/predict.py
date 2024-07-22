import joblib
import numpy as np
from tensorflow.keras.preprocessing import image


def load_and_preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# Load the trained model
model = joblib.load("elephant_sound_classifier.joblib")

# Path to the new spectrogram image
new_image_path = r"E:\SOUNDCLASSIFICATIONCNN\scripts\rumble_8859.8440_8861.5440.png"

# Preprocess the image
img_array = load_and_preprocess_image(new_image_path)

# Make prediction
prediction = model.predict(img_array)
print("Prediction:", prediction)

if prediction[0][0] > 0.5:
    print("The sound is NOT an elephant sound.")
else:
    print("The sound is an elephant sound.")

# # r"E:\SOUNDCLASSIFICATIONCNN\scripts\mixkit-angry-dragon-growl-309.png"-not elephant
# E:\SOUNDCLASSIFICATIONCNN\scripts\rumble_8859.8440_8861.5440.png"-elephant
