import os
import matplotlib.pyplot as plt
import numpy as np

# Import the functions from our other scripts
from data_loader import load_eeg_data
from preprocess import filter_data, create_epochs
from encode import rate_coding

# --- Configuration ---
# --- CORRECTED LINE ---
DATASET_PATH = os.path.join('data', 'ds004504')
# ----------------------
SUBJECT_ID = 'sub-001' # We'll stick with a subject that has good data
EPOCH_DURATION = 2.0   # Length of each data chunk in seconds

def main():
    """Main function to run the data processing pipeline."""
    
    print("--- Starting EEG Processing Pipeline ---")

    # 1. Load Data
    print(f"\nStep 1: Loading data for {SUBJECT_ID}...")
    raw = load_eeg_data(DATASET_PATH, SUBJECT_ID)
    
    if raw is None:
        print("Halting pipeline due to data loading error.")
        return

    # 2. Preprocess Data
    print("\nStep 2: Preprocessing data...")
    raw_filtered = filter_data(raw.copy())
    epochs = create_epochs(raw_filtered, duration=EPOCH_DURATION)
    
    # Get data as a NumPy array for encoding
    eeg_data_np = epochs.get_data()
    sfreq = epochs.info['sfreq']

    # 3. Encode Data into Spikes
    print("\nStep 3: Encoding data into spikes...")
    spike_data = rate_coding(eeg_data_np, sfreq=sfreq)
    
    print("\n--- Pipeline Finished Successfully ---")

    # 4. Corrected Visualization
    epoch_to_plot = 10 
    channels_to_plot = range(5)
    num_channels = len(channels_to_plot)

    fig, axes = plt.subplots(num_channels + 1, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f'EEG Signals and Encoded Spikes (Epoch {epoch_to_plot})', fontsize=16)

    # Plot each EEG channel on its own subplot
    for i, ch_idx in enumerate(channels_to_plot):
        axes[i].plot(epochs.times, eeg_data_np[epoch_to_plot, ch_idx, :])
        axes[i].set_ylabel(f'Ch {ch_idx}')
        axes[i].set_yticks([])

    # Plot the resulting spikes on the last subplot
    spike_axes = axes[-1]
    for i, ch_idx in enumerate(channels_to_plot):
        spike_times = epochs.times[spike_data[epoch_to_plot, ch_idx, :] == 1]
        spike_axes.vlines(spike_times, ymin=i - 0.4, ymax=i + 0.4, color='black')
    
    spike_axes.set_xlabel('Time (s)')
    spike_axes.set_ylabel('Channels')
    spike_axes.set_ylim(-0.5, num_channels - 0.5)
    spike_axes.set_yticks(channels_to_plot)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()

    # --- MOVED THESE LINES INSIDE THE FUNCTION ---
    # Save the processed data for the next steps
    output_filename = f'{SUBJECT_ID}_spike_data.npy'
    np.save(output_filename, spike_data)
    print(f"\nSpike data saved to {output_filename}")


if _name_ == '_main_':
    main()