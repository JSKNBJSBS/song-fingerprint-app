import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
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
        fig, ax = plt.subplots(3, 1, figsize=(8, 12))
        
        # A. Spectrogram
        f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=512)
        ax[0].pcolormesh(t, f, 10 * np.log10(Sxx))
        ax[0].set_title("Spectrogram")
        
        # B. Constellation (Placeholder - ensure these variables exist in your code)
        ax[1].set_title("Constellation of Peaks")
        
        # C. Offset Histogram
        ax[2].set_title("Offset Histogram")
        
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
