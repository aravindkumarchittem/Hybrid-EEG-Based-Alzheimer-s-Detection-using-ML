import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import EEGSpikeDataset
from model import CSNN
from tqdm import tqdm

# ---------------- Configuration ----------------
PROCESSED_DATA_DIR = r"C:\Users\aravi\Desktop\snn_alzheimer_detection\data\processed_spikes"
MODEL_PATH = r"C:\Users\aravi\Desktop\snn_alzheimer_detection\snn_alzheimer_model.pth"

BATCH_SIZE = 4
EPOCHS = 70
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2

# ---------------- Device Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# ---------------- Dataset ----------------
if not os.path.exists(PROCESSED_DATA_DIR):
    raise FileNotFoundError(f"❌ Processed data folder not found: {PROCESSED_DATA_DIR}")

dataset = EEGSpikeDataset(processed_dir=PROCESSED_DATA_DIR)
if len(dataset) == 0:
    raise ValueError("❌ No EEG spike data found in the processed folder. Run encode.py first!")

train_size = int((1 - VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Loaded dataset with {len(dataset)} samples ({train_size} train / {val_size} val)")

# ---------------- Model Setup ----------------
num_channels = dataset[0][0].shape[1]  # EEG channels per sample
model = CSNN(num_channels=num_channels, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------- Resume if available ----------------
best_val_acc = 0.0
if os.path.exists(MODEL_PATH):
    print(f"📂 Found existing model at {MODEL_PATH}. Loading weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Loaded previous model checkpoint.")

# ---------------- Training Loop ----------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for data, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Training"):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)

    train_loss /= len(train_loader.dataset)

    # ---------------- Validation ----------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} Validation"):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total

    print(f"\n📊 Epoch {epoch}/{EPOCHS}")
    print(f"   🔹 Train Loss: {train_loss:.4f}")
    print(f"   🔹 Val Loss:   {val_loss:.4f}")
    print(f"   🔹 Val Acc:    {val_acc:.4f}\n")

    # ---------------- Save Best Model ----------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ Saved new best model (Val Acc={best_val_acc:.4f}) → {MODEL_PATH}")

print(f"\n🎯 Training complete. Best validation accuracy: {best_val_acc:.4f}")
