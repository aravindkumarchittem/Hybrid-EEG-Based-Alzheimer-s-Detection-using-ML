
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from dataset import EEGSpikeDataset
from model import CSNN

# ---------------- Configuration ----------------
PROCESSED_DATA_DIR = r"C:\Users\aravi\Desktop\snn_alzheimer_detection\data\processed_spikes"
MODEL_PATH = r"C:\Users\aravi\Desktop\snn_alzheimer_detection\snn_alzheimer_model.pth"
OUTPUT_PATH = r"C:\Users\aravi\Desktop\snn_alzheimer_detection\evaluation_results.csv"

BATCH_SIZE = 4

# ---------------- Device Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using {device} for evaluation")

# ---------------- Dataset ----------------
if not os.path.exists(PROCESSED_DATA_DIR):
    raise FileNotFoundError(f"❌ Processed data folder not found: {PROCESSED_DATA_DIR}")

dataset = EEGSpikeDataset(processed_dir=PROCESSED_DATA_DIR)
if len(dataset) == 0:
    raise ValueError("❌ No EEG spike data found in the processed folder. Run encode.py first!")

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"✅ Loaded dataset with {len(dataset)} samples")

# ---------------- Model Setup ----------------
num_channels = dataset[0][0].shape[1]
model = CSNN(num_channels=num_channels, num_classes=2).to(device)

# 🧩 Initialize FC layers with a dummy forward pass
dummy_data, _ = dataset[0]
dummy_data = dummy_data.unsqueeze(0).to(device)
_ = model(dummy_data)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Trained model not found at {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Model loaded successfully from {MODEL_PATH}")

# ---------------- Evaluation ----------------
all_preds = []
all_labels = []

with torch.no_grad():
    for data, labels in tqdm(data_loader, desc="Evaluating"):
        data = data.to(device)
        outputs = model(data)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ---------------- Metrics ----------------
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
rec = recall_score(all_labels, all_preds, average="binary", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print("\n📊 --- Final Model Performance ---")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print("\nConfusion Matrix:")
print(cm)

# ---------------- Save Results ----------------
results = pd.DataFrame({
    "true_label": all_labels,
    "predicted_label": all_preds
})
results.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Evaluation results saved to: {OUTPUT_PATH}")
