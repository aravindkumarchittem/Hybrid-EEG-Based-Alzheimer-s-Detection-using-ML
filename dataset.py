import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EEGSpikeDataset(Dataset):
    def __init__(self, processed_dir):
        """
        Args:
            processed_dir (str): Path to folder containing .npy spike files and labels.csv
        """
        self.processed_dir = processed_dir
        labels_path = os.path.join(processed_dir, 'labels.csv')

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.csv not found in {processed_dir}. Run encode.py first!")

        # Load CSV containing file names and labels
        self.labels_df = pd.read_csv(labels_path)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Get row info
        row = self.labels_df.iloc[idx]
        file_path = os.path.join(self.processed_dir, row['file_name'])
        label = int(row['label'])

        # Load .npy spike data
        spike_data = np.load(file_path)

        # Convert numpy array to torch tensor
        # Shape expected: (n_epochs, n_channels, n_times)
        spike_tensor = torch.tensor(spike_data, dtype=torch.float32)

        # You can flatten or average epochs depending on model input
        # Example: average across epochs for simplicity
        spike_tensor = spike_tensor.mean(dim=0)  # (n_channels, n_times)

        return spike_tensor, torch.tensor(label, dtype=torch.long)


