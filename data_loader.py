import mne
import os

def load_eeg_data(data_path, subject_id):
    """
    Loads a preprocessed EEG file for a specific subject.

    Args:
        data_path (str): The base path to the 'ds004504' dataset directory.
        subject_id (str): The subject identifier (e.g., 'sub-001').

    Returns:
        mne.io.Raw: The loaded raw EEG data object.
    """
    # Construct the file path to the preprocessed EEG data
    # We use the '.set' files from the derivatives folder
    eeg_file_path = os.path.join(
        data_path,
        'derivatives',
        subject_id,
        'eeg',
        f'{subject_id}_task-eyesclosed_eeg.set'
    )

    try:
        # Load the data using MNE's function for EEGLAB '.set' files
        raw = mne.io.read_raw_eeglab(eeg_file_path, preload=True)
        print(f"Successfully loaded data for {subject_id}")
        return raw
    except FileNotFoundError:
        print(f"Error: Data file not found for {subject_id} at {eeg_file_path}")
        return None

if _name_ == '_main_':
    # This block is for testing the function directly
    # It will only run when you execute python 01_data_loader.py
    
    # Define the path to your dataset and a subject to test
    DATASET_PATH = os.path.join('data', 'ds004504')
    TEST_SUBJECT = 'sub-001' # Change this to test other subjects

    # Load the data
    raw_data = load_eeg_data(DATASET_PATH, TEST_SUBJECT)

    # If data is loaded, print some information
    if raw_data:
        print("\n--- Data Info ---")
        print(raw_data.info)
        
        # Plot the first 5 seconds of data for a quick look
        raw_data.plot(duration=5, n_channels=19, scalings='auto')