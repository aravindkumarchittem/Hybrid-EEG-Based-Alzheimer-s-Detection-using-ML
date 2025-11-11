


import os
import pandas as pd
import numpy as np

# Import helper functions
from data_loader import load_eeg_data
from preprocess import filter_data, create_epochs
from encode import rate_coding

# --- Configuration ---
DATASET_PATH = os.path.join('data', 'ds004504')
OUTPUT_DIR = os.path.join('data', 'processed_spikes')
EPOCH_DURATION = 2.0

def process_all_subjects():
    """
    Processes the EEG data for all relevant subjects (A & C),
    encodes it into spike format, and saves labels.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    participants_file = os.path.join(DATASET_PATH, 'participants.tsv')
    if not os.path.exists(participants_file):
        print(f"❌ Error: participants.tsv not found at {participants_file}")
        return
        
    participants = pd.read_csv(participants_file, sep='\t')
    diagnosis_column_name = 'Group'
    
    # Only Alzheimer's (A) and Controls (C)
    subjects_to_process = participants[participants[diagnosis_column_name].isin(['A', 'C'])]
    print(f"✅ Found {len(subjects_to_process)} subjects (Groups: A, C)")

    metadata = []

    for index, subject in subjects_to_process.iterrows():
        subject_id = subject['participant_id']
        group = subject[diagnosis_column_name]
        print(f"\n--- Processing {subject_id} (Group: {group}) ---")
        
        raw = load_eeg_data(DATASET_PATH, subject_id)
        if raw is None:
            print(f"⚠️ Skipping {subject_id} (could not load data)")
            continue
            
        raw_filtered = filter_data(raw.copy())
        epochs = create_epochs(raw_filtered, duration=EPOCH_DURATION)
        eeg_data_np = epochs.get_data()
        
        spike_data = rate_coding(eeg_data_np, sfreq=epochs.info['sfreq'])
        
        output_filename = f'{subject_id}_spike_data.npy'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        np.save(output_path, spike_data)
        
        # ✅ Correct label mapping
        label = 0 if group == 'A' else 1  # 0: Alzheimer, 1: Healthy
        metadata.append([output_filename, label])

    # Save metadata
    metadata_df = pd.DataFrame(metadata, columns=['filename', 'label'])
    labels_csv_path = os.path.join(OUTPUT_DIR, 'labels.csv')
    metadata_df.to_csv(labels_csv_path, index=False)
    
    print(f"\n✅ Processing complete. Processed {len(metadata)} subjects.")
    print(f"📁 Spike data saved in: {OUTPUT_DIR}")
    print(f"📄 Labels file: {labels_csv_path}")

if __name__ == "__main__":
    process_all_subjects()

