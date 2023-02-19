import glob
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

path = "audio data/*.mp3"
all_files = glob.glob(path)

for file in all_files:
    y, sr = librosa.load(file, sr=None)
    # Perform processing on the audio file
    # Extract melspectrogram features using librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    # Convert the power spectrogram to decibels
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Display the melspectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()
