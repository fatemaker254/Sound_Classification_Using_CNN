import joblib
import numpy as np
import tensorflow as tf


def load_model(model_path):
    return joblib.load(model_path)


def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def predict_image(model, image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    return "Elephant" if prediction[0] > 0.5 else "Not Elephant"


# Example usage
if __name__ == "__main__":
    model_path = "elephant_sound_classifier.joblib"
    new_image_path = (
        r"E:\SOUNDCLASSIFICATIONCNN\dataset\elephant\rumble_8585.3320_8592.1670.png"
    )
    # new_image_path = r"E:\SOUNDCLASSIFICATIONCNN\scripts\rumble_8859.8440_8861.5440.png"
    model = load_model(model_path)
    result = predict_image(model, new_image_path)
    print(f"The sound is: {result}")
