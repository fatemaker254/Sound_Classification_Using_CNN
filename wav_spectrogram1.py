import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt


# Function to create a spectrogram from an audio file and save it as an image
def create_spectrogram(audio_file, image_file):
    """
    Create a spectrogram from an audio file and save it as an image.

    Parameters:
    audio_file (str): Path to the input audio file (.wav).
    image_file (str): Path to the output image file (.png).
    """
    # Create a new figure
    fig = plt.figure(figsize=[10, 4])
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)  # Use the default sampling rate

    # Create a mel-spectrogram
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    # Display the spectrogram
    librosa.display.specshow(log_ms, sr=sr, x_axis="time", y_axis="mel")
    plt.axis("off")  # Hide axes for cleaner image

    # Save the figure
    fig.savefig(image_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # Close the figure to free memory


# Function to convert all .wav files in the input directory to .png spectrogram images
def create_pngs_from_wavs(input_path, output_path):
    """
    Convert all .wav files in the input directory to .png spectrogram images
    and save them to the output directory.

    Parameters:
    input_path (str): Path to the directory containing input .wav files.
    output_path (str): Path to the directory to save output .png files.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # List all files in the input directory
    files = os.listdir(input_path)

    # Process each file
    for file in files:
        if file.endswith(".wav"):
            input_file = os.path.join(input_path, file)
            output_file = os.path.join(output_path, file.replace(".wav", ".png"))
            create_spectrogram(input_file, output_file)


# Example usage
input_path = r"E:\SOUNDCLASSIFICATIONCNN\dataset\NonElephantSounds\audio files"  #  path to .wav files
output_path = r"E:\SOUNDCLASSIFICATIONCNN\dataset\nonelephantsoundspectogram"  #  path to save .png files
create_pngs_from_wavs(input_path, output_path)
