import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_spectrogram(audio_path, output_path):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Compute the Short-Time Fourier Transform (STFT)
        D = librosa.stft(y)
        
        # Convert the complex-valued STFT matrix to a magnitude spectrogram
        S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
        
        # Plot the spectrogram
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        
        # Save the spectrogram as an image file
        plt.savefig(output_path)
        plt.close()
        print(f"Spectrogram saved for {audio_path} at {output_path}")
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

def process_audio_files(input_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if os.path.isdir(input_path):
        # Process all .mp3 files in the directory
        for audio_file in os.listdir(input_path):
            if audio_file.endswith(".mp3"):  # Ensure the file is a .mp3 file
                audio_path = os.path.join(input_path, audio_file)
                output_path = os.path.join(output_folder, os.path.splitext(audio_file)[0] + '.png')
                create_spectrogram(audio_path, output_path)
    elif input_path.endswith(".mp3"):
        # Process a single .mp3 file
        audio_path = input_path
        output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(audio_path))[0] + '.png')
        create_spectrogram(audio_path, output_path)
    else:
        print("The input path must be a directory or a .mp3 file")

if __name__ == "__main__":
    input_path = r"C:\Users\ZAZ\Desktop\elp"  # Update this to your audio file or folder path
    output_folder = r"C:\Users\ZAZ\Desktop\elp\output"  # Update this to your output folder path
    
    process_audio_files(input_path, output_folder)
