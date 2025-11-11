# Hybrid-EEG-Based-Alzheimer-s-Detection-using-ML
A hybrid deep learning project that combines Spiking Neural Network (SNN)-inspired rate coding and a 2D Convolutional Neural Network (CNN) to detect Alzheimer’s Disease (AD) from EEG signals.


⚙️ Requirements

Python 3.9+

Install all dependencies:
pip install -r requirements.txt


requirements.txt
mne
numpy
pandas
torch
matplotlib
scikit-learn
tqdm


🧾 Dataset Setup

Download dataset from OpenNeuro:
👉 https://openneuro.org/datasets/ds004504/versions/1.0.8

Extract it inside your project folder like this:
data/ds004504/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set


📂 Folder Structure
Hybrid-Alzheimer-Detection/
├── data/ds004504/            ← Raw EEG dataset
├── data/processed_spikes/    ← Encoded .npy + labels.csv
├── encode.py                 ← Preprocess & encode EEG
├── dataset.py                ← PyTorch dataset loader
├── model.py                  ← 2D CNN architecture
├── train.py                  ← Model training
├── evaluate.py               ← Model evaluation
├── predict.py                ← Final predictions
└── README.md



▶️ Running the Pipeline
Step 1️⃣ – Encode EEG Data
Convert all .set EEG files into spike data and labels:
python encode.py

Output:
.npy files for each subject (e.g., sub-001.npy)
labels.csv file inside data/processed_spikes/

Step 2️⃣ – Train the CNN Model
Train and validate the hybrid CNN classifier:
python train.py

Output:
snn_alzheimer_model.pth → best-performing trained model

Step 3️⃣ – Evaluate Model Performance
Check scientific metrics (accuracy, precision, recall):
python evaluate.py

Example Output:

Accuracy: 93.85%
Precision: 90.0%
Recall: 100%

Step 4️⃣ – Run Predictions
Generate Alzheimer’s/Healthy predictions for each subject:
python predict.py

Example Output:

Subject sub-001 → Alzheimer’s (confidence = 87.9%)
Subject sub-037 → Healthy (confidence = 91.3%)



Run these commands in order:

python encode.py      # Convert EEG → spike data
python train.py       # Train CNN model
python evaluate.py    # Evaluate metrics
python predict.py     # Get Alzheimer’s/Healthy prediction


📊 Performance Summary
| Metric         | Score                             |
| -------------- | --------------------------------- |
| Accuracy       | 93.85 %                           |
| Precision (AD) | 90.0 %                            |
| Recall (AD)    | 100 %                             |
| Dataset Size   | 65 subjects                       |
| Model Type     | Hybrid SNN (Rate Coding) + 2D CNN |



🧩 Key Highlights

Fully automated modular pipeline (encode → train → evaluate → predict).
Reproducible results with same parameters and structure.
Non-invasive, interpretable, and biologically inspired approach.
Designed for EEG-based neurological screening research.


🧠 Summary

EEG → filtered (1–45 Hz), epoch = 2 s
Converted to spike trains (rate coding)
Averaged into 2D “spike images” (19 × 500)
Trained CNN → classifies AD vs Healthy


📜 References
OpenNeuro (2023) — EEG dataset: Alzheimer’s Disease, FTD, and Healthy subjects (ds004504).
Gramfort, A. et al. (2013) — MNE-Python for EEG/MEG analysis.
Ieracitano, C. et al. (2020) — Explainable ML for EEG-based Alzheimer’s detection.
Paszke, A. et al. (2019) — PyTorch: Deep learning library.
Kulkarni, S. R. & Raj, B. (2018) — Rate coding in Spiking Neural Networks.


