import librosa
import numpy as np
import pickle
import os

def get_fingerprint(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    spec = np.abs(librosa.stft(y))
    return np.mean(spec, axis=1)

def identify_song(file_path, db_path='song_database.pkl'):
    """
    Loads the song database, computes the fingerprint of the query file,
    and returns the filename (without extension) of the closest match.
    """
    if not os.path.exists(db_path):
        return "Database not found"

    with open(db_path, 'rb') as f:
        db = pickle.load(f)

    query_fp = get_fingerprint(file_path)

    best_match = None
    best_score = -1  # higher cosine similarity is better

    for song_name, song_fp in db.items():
        # Make sure vectors are same length (truncate to shorter one)
        min_len = min(len(query_fp), len(song_fp))
        q = query_fp[:min_len]
        s = song_fp[:min_len]

        # Cosine similarity
        denom = (np.linalg.norm(q) * np.linalg.norm(s))
        score = np.dot(q, s) / denom if denom != 0 else 0

        if score > best_score:
            best_score = score
            best_match = song_name

    if best_match is None:
        return "No match found"

    # Strip extension, per the submission format requirement
    return os.path.splitext(best_match)[0]
