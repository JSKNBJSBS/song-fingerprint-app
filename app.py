import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
# Replace 'fingerprint' with the actual name of your file containing the functions
from fingerprint import identify_song 

st.title("🎵 Audio Fingerprint Matcher")

# 1. Load Database
if os.path.exists('song_database.pkl'):
    with open('song_database.pkl', 'rb') as f:
        db = pickle.load(f)
else:
    st.error("Database not found. Please index your songs first.")
    st.stop()

# 2. Tabs for Modes
tab1, tab2 = st.tabs(["Single-Clip Mode", "Batch Mode"])

with tab1:
    uploaded_file = st.file_uploader("Upload a clip to identify", type=["wav"])
    if uploaded_file:
        with open("query.wav", "wb") as f: f.write(uploaded_file.getbuffer())
        st.subheader("Intermediate Processing Steps")

        # Audio processing
        data = np.frombuffer(uploaded_file.getbuffer(), dtype=np.int16)
        fs = 22050

        # Plots
        fig, ax = plt.subplots(3, 1, figsize=(8, 14))

        # A. Spectrogram
        f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=512)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        ax[0].pcolormesh(t, f, Sxx_db, shading='auto')
        ax[0].set_title("Spectrogram")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Frequency (Hz)")

        # B. Constellation - find local peaks in the spectrogram
        neighborhood_size = 10
        local_max = maximum_filter(Sxx_db, size=neighborhood_size) == Sxx_db
        threshold = np.percentile(Sxx_db, 95)  # keep only strongest peaks
        peaks_mask = local_max & (Sxx_db > threshold)
        peak_freq_idx, peak_time_idx = np.where(peaks_mask)
        peak_times = t[peak_time_idx]
        peak_freqs = f[peak_freq_idx]

        ax[1].scatter(peak_times, peak_freqs, s=8, c='red')
        ax[1].set_title("Constellation of Peaks")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Frequency (Hz)")
        ax[1].set_xlim(t.min(), t.max())
        ax[1].set_ylim(f.min(), f.max())

        # C. Offset Histogram - pairwise time differences between peaks
        if len(peak_times) > 1:
            offsets = []
            for i in range(len(peak_times)):
                for j in range(i + 1, min(i + 10, len(peak_times))):
                    offsets.append(peak_times[j] - peak_times[i])
            ax[2].hist(offsets, bins=30, color='steelblue')
        ax[2].set_title("Offset Histogram")
        ax[2].set_xlabel("Time Offset (s)")
        ax[2].set_ylabel("Count")

        plt.tight_layout()
        st.pyplot(fig)

        # Identify
        prediction = identify_song("query.wav")
        st.success(f"Match Found: {prediction}")

with tab2:
    st.subheader("Batch Mode")
    batch_files = st.file_uploader("Upload multiple clips", accept_multiple_files=True)
    if st.button("Generate results.csv"):
        results = []
        for f in batch_files:
            # Save temporary file for each
            with open("temp.wav", "wb") as tmp: tmp.write(f.getbuffer())
            prediction = identify_song("temp.wav")
            results.append({'filename': f.name, 'prediction': prediction})

        df = pd.DataFrame(results)
        csv = df.to_csv(index=False)
        st.download_button("Download results.csv", csv, "results.csv", "text/csv")
