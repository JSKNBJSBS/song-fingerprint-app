import os
import pickle
from fingerprint import get_fingerprint

# The base folder is 'songs'
base_folder = 'songs'
# The actual songs are in this subfolder
sub_folder = 'EE200 Project Song Database'
folder_path = os.path.join(base_folder, sub_folder)

db = {}

if not os.path.exists(folder_path):
    print(f"❌ ERROR: Could not find folder at {folder_path}")
else:
    print(f"Indexing files in: {folder_path}...")
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.mp3', '.wav')):
            path = os.path.join(folder_path, file)
            db[file] = get_fingerprint(path)
            print(f"Indexed: {file}")
    
    with open('song_database.pkl', 'wb') as f:
        pickle.dump(db, f)
    print("✅ SUCCESS: song_database.pkl created.")
