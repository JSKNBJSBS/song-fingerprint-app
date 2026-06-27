import librosa
import numpy as np

def get_fingerprint(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    spec = np.abs(librosa.stft(y))
    return np.mean(spec, axis=1)
