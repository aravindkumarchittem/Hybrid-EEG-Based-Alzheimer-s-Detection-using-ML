import mne

def filter_data(raw, l_freq=1.0, h_freq=45.0):
    """
    Applies a band-pass filter to the raw EEG data.

    Args:
        raw (mne.io.Raw): The raw EEG data object.
        l_freq (float): The lower frequency bound.
        h_freq (float): The upper frequency bound.

    Returns:
        mne.io.Raw: The filtered raw EEG data object.
    """
    raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
    print(f"Applied a band-pass filter between {l_freq} Hz and {h_freq} Hz.")
    return raw

def create_epochs(raw, duration=2.0, overlap=0.5):
    """
    Segments the continuous data into epochs.

    Args:
        raw (mne.io.Raw): The raw EEG data object.
        duration (float): The duration of each epoch in seconds.
        overlap (float): The overlap between epochs in seconds.

    Returns:
        mne.Epochs: The created epochs object.
    """
    epochs = mne.make_fixed_length_epochs(
        raw, 
        duration=duration, 
        overlap=overlap, 
        preload=True,
        verbose=False
    )
    print(f"Created {len(epochs)} epochs of {duration}s each with {overlap}s overlap.")
    return epochs