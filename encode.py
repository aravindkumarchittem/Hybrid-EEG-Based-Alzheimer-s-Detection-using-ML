import os
import numpy as np
import mne
import pandas as pd
from tqdm import tqdm
import glob

# ---------------- Configuration ----------------
raw_data_dir = r'C:\Users\aravi\Desktop\snn_alzheimer_detection\data\ds004504\derivatives'  # BIDS derivatives folder
processed_data_dir = r'C:\Users\aravi\Desktop\snn_alzheimer_detection\data\processed_spikes'  # Where spike files will be saved
sfreq = 250  # EEG sampling frequency
max_firing_rate = 100  # Max firing rate for rate coding

# ---------------- Label Mapping ----------------
def get_label(subj_id):
    sub_num = int(subj_id.split('-')[1])
    if 1 <= sub_num <= 36:
        return 0  # Alzheimer's
    elif 37 <= sub_num <= 65:
        return 1  # Healthy Control
    else:
        return None  # FTD subjects (ignored)

# Create output folder if missing
os.makedirs(processed_data_dir, exist_ok=True)

# ---------------- Spike Encoding ----------------
def rate_coding(epochs_data, max_firing_rate=100, sfreq=250):
    min_val = epochs_data.min()
    max_val = epochs_data.max()
    if max_val - min_val > 0:
        normalized_data = (epochs_data - min_val) / (max_val - min_val)
    else:
        normalized_data = epochs_data

    firing_rates = normalized_data * max_firing_rate
    dt = 1.0 / sfreq
    spike_trains = (np.random.rand(*epochs_data.shape) < (firing_rates * dt)).astype(int)
    return spike_trains

# ---------------- Process EEG Files ----------------
print("Starting spike encoding for all EEG files...")

eeg_files = glob.glob(os.path.join(raw_data_dir, '**', '*.set'), recursive=True)
print(f"Found {len(eeg_files)} EEG files in '{raw_data_dir}'")

if len(eeg_files) == 0:
    print("⚠️ No .set files found. Check your dataset path.")
else:
    print("Example files:")
    for f in eeg_files[:5]:
        print(" →", f)

labels_list = []

for file_path in tqdm(eeg_files, desc="Encoding EEG files"):
    file_name = os.path.basename(file_path)
    subj_id = file_name.split('_')[0]
    label = get_label(subj_id)

    if label is None:
        print(f"⚠️ Skipping {subj_id} (FTD subject not used)")
        continue

    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)

        events = mne.make_fixed_length_events(raw, duration=2.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True, verbose=False)
        data = epochs.get_data()

        spike_data = rate_coding(data, max_firing_rate=max_firing_rate, sfreq=sfreq)

        out_file = os.path.join(processed_data_dir, f"{subj_id}.npy")
        np.save(out_file, spike_data)
        labels_list.append([f"{subj_id}.npy", label])

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")

# Save labels.csv
if labels_list:
    labels_df = pd.DataFrame(labels_list, columns=['file_name', 'label'])
    labels_df.to_csv(os.path.join(processed_data_dir, 'labels.csv'), index=False)
    print(f"\n✅ Spike encoding complete. Processed {len(labels_list)} subjects.")
    print(f"📁 Data saved in: {processed_data_dir}")
else:
    print("\n⚠️ No EEG files were processed. Check paths or dataset structure.")
