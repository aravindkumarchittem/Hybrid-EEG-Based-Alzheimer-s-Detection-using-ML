import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import CSNN

# ---------------- Configuration ----------------
PROCESSED_DATA_DIR = r"C:\Users\aravi\Desktop\snn_alzheimer_detection\data\processed_spikes"
MODEL_PATH = r"C:\Users\aravi\Desktop\snn_alzheimer_detection\snn_alzheimer_model.pth"
OUTPUT_PATH = r"C:\Users\aravi\Desktop\snn_alzheimer_detection\evaluation_results.csv"

BATCH_SIZE = 4
LABEL_MAP = {0: "Alzheimer", 1: "Healthy"} # From encode.py: 0=Alzheimer, 1=Healthy

# ---------------- Dataset ----------------
# This internal Dataset class is fine, it loads and averages epochs
class ProcessedSpikeDataset(Dataset):
    def __init__(self, data_dir):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
        if not files:
            raise ValueError(f"No .npy files found in {data_dir}")
        self.files = [os.path.join(data_dir, f) for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = np.load(path)
        if arr.ndim == 3:
            # This is the key logic from your dataset.py:
            # Average across epochs to get a 2D array
            arr = arr.mean(axis=0)
        elif arr.ndim != 2:
            arr = np.squeeze(arr)
            if arr.ndim == 3:
                arr = arr.mean(axis=0)
        arr = arr.astype(np.float32)
        return arr, os.path.basename(path)

# ---------------- Main ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")

    ds = ProcessedSpikeDataset(PROCESSED_DATA_DIR)
    # Use the default collate_fn, as the dataset returns simple tensors
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    sample_arr, _ = ds[0]
    num_channels = sample_arr.shape[0]
    print(f"✅ Found {len(ds)} files; each sample has {num_channels} channels")

    model = CSNN(num_channels=num_channels, num_classes=2).to(device)

    # Dummy forward for model init
    model.eval()
    with torch.no_grad():
        dummy = torch.tensor(sample_arr, dtype=torch.float32).unsqueeze(0).to(device)
        _ = model(dummy)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # --- SIMPLIFIED AND CORRECTED MODEL LOADING ---
    # Your train.py saves model.state_dict(), so we load it directly.
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Model loaded successfully.\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("This might be due to a mismatch between your model.py and the saved .pth file.")
        print("Please ensure you have trained a new model with the latest model.py script.")
        return

    # ---- Prediction loop ----
    results = []
    model.eval()
    with torch.no_grad():
        # The DataLoader now returns a batch of [data, fnames]
        for data_batch, fnames_batch in tqdm(loader, desc="Predicting"):
            
            # --- NORMALIZATION REMOVED ---
            # The data is fed to the model exactly as it was during training.
            x = data_batch.float().to(device)
            
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            for fname, pred, prob in zip(fnames_batch, preds, probs):
                label_text = LABEL_MAP[int(pred)]
                confidence = float(prob[int(pred)]) * 100.0
                results.append((fname, label_text, confidence))

    df = pd.DataFrame(results, columns=["filename", "predicted_label", "confidence_percent"])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📁 Predictions saved to: {OUTPUT_PATH}")

    # ---- Summary ----
    alz_df = df[df['predicted_label'] == "Alzheimer"]
    healthy_df = df[df['predicted_label'] == "Healthy"]

    print(f"\n🧠 Prediction Summary (total {len(df)} samples):")
    print(f"  🔴 Alzheimer: {len(alz_df)} samples")
    print(f"  🟢 Healthy: {len(healthy_df)} samples")

    print("\nList of Alzheimer files with confidence:")
    for _, row in alz_df.iterrows():
        print(f"  {row['filename']} — {row['confidence_percent']:.2f}%")

    print("\nList of Healthy files with confidence:")
    for _, row in healthy_df.iterrows():
        print(f"  {row['filename']} — {row['confidence_percent']:.2f}%")

if __name__ == "__main__":
    main()



# import os
# import torch
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from model import CSNN

# # ---------------- Configuration ----------------
# PROCESSED_DATA_DIR = r"C:\Users\aravi\Desktop\snn_alzheimer_detection\data\processed_spikes"
# MODEL_PATH = r"C:\Users\aravi\Desktop\snn_alzheimer_detection\snn_alzheimer_model.pth"
# OUTPUT_PATH = r"C:\Users\aravi\Desktop\snn_alzheimer_detection\evaluation_results.csv"
# # PROCESSED_DATA_DIR = r"D:\SNN\snn_alzheimer_detection\data\processed_spikes"
# # MODEL_PATH = r"D:\SNN\snn_alzheimer_detection\snn_alzheimer_model.pth"
# # OUTPUT_PATH = r"D:\SNN\snn_alzheimer_detection\predicted_labels.csv"

# BATCH_SIZE = 4
# LABEL_MAP = {0: "Alzheimer", 1: "Healthy"}

# # ---------------- Dataset ----------------
# class ProcessedSpikeDataset(Dataset):
#     def __init__(self, data_dir):
#         if not os.path.exists(data_dir):
#             raise FileNotFoundError(f"Data directory not found: {data_dir}")
#         files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
#         if not files:
#             raise ValueError(f"No .npy files found in {data_dir}")
#         self.files = [os.path.join(data_dir, f) for f in files]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path = self.files[idx]
#         arr = np.load(path)
#         if arr.ndim == 3:
#             arr = arr.mean(axis=0)
#         elif arr.ndim != 2:
#             arr = np.squeeze(arr)
#             if arr.ndim == 3:
#                 arr = arr.mean(axis=0)
#         arr = arr.astype(np.float32)
#         return arr, os.path.basename(path)

# # ---------------- Helpers ----------------
# def normalize_sample(x: np.ndarray) -> np.ndarray:
#     mu = x.mean()
#     sigma = x.std() + 1e-8
#     return (x - mu) / sigma

# # ---------------- Main ----------------
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"⚙️ Using device: {device}")

#     ds = ProcessedSpikeDataset(PROCESSED_DATA_DIR)
#     loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: b)

#     sample_arr, _ = ds[0]
#     num_channels = sample_arr.shape[0]
#     print(f"✅ Found {len(ds)} files; each sample has {num_channels} channels")

#     model = CSNN(num_channels=num_channels, num_classes=2).to(device)

#     # Dummy forward for model init
#     model.eval()
#     with torch.no_grad():
#         dummy = torch.tensor(sample_arr, dtype=torch.float32).unsqueeze(0).to(device)
#         _ = model(dummy)

#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

#     ckpt = torch.load(MODEL_PATH, map_location=device)
#     if isinstance(ckpt, dict) and any(k.startswith('epoch') or 'optimizer' in k for k in ckpt.keys()):
#         if 'state_dict' in ckpt:
#             ckpt = ckpt['state_dict']
#     try:
#         model.load_state_dict(ckpt)
#     except RuntimeError:
#         model_dict = model.state_dict()
#         filtered = {k: v for k, v in ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
#         model_dict.update(filtered)
#         model.load_state_dict(model_dict)
#         print("⚠️ Loaded partially matching weights (some layers ignored).")

#     print("✅ Model loaded successfully.\n")

#     # ---- Prediction loop ----
#     results = []
#     model.eval()
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Predicting"):
#             arrays = [normalize_sample(item[0]) for item in batch]
#             fnames = [item[1] for item in batch]

#             x = torch.stack([torch.tensor(a, dtype=torch.float32) for a in arrays], dim=0).to(device)
#             outputs = model(x)
#             probs = torch.softmax(outputs, dim=1).cpu().numpy()
#             preds = probs.argmax(axis=1)

#             for fname, pred, prob in zip(fnames, preds, probs):
#                 label_text = LABEL_MAP[int(pred)]
#                 confidence = float(prob[int(pred)]) * 100.0
#                 results.append((fname, label_text, confidence))

#     df = pd.DataFrame(results, columns=["filename", "predicted_label", "confidence_percent"])
#     df.to_csv(OUTPUT_PATH, index=False)
#     print(f"\n📁 Predictions saved to: {OUTPUT_PATH}")

#     # ---- Summary ----
#     alz_df = df[df['predicted_label'] == "Alzheimer"]
#     healthy_df = df[df['predicted_label'] == "Healthy"]

#     print(f"\n🧠 Prediction Summary (total {len(df)} samples):")
#     print(f"  🔴 Alzheimer: {len(alz_df)} samples")
#     print(f"  🟢 Healthy: {len(healthy_df)} samples")

#     print("\nList of Alzheimer files with confidence:")
#     for _, row in alz_df.iterrows():
#         print(f"  {row['filename']} — {row['confidence_percent']:.2f}%")

#     print("\nList of Healthy files with confidence:")
#     for _, row in healthy_df.iterrows():
#         print(f"  {row['filename']} — {row['confidence_percent']:.2f}%")

# if __name__ == "__main__":
#     main()
