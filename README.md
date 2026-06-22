# 🛡️ NetGuard IDS — Real-Time Network Intrusion Detection System
### Using CICIDS2017 Dataset + Bidirectional LSTM Neural Network

---

## 📁 PROJECT FOLDER STRUCTURE

```
IDS_Project/
│
├── app.py                ← Flask web application (run this last)
├── preprocess.py         ← Step 1: Data cleaning & preparation
├── train_model.py        ← Step 2: LSTM model training
├── requirements.txt      ← All Python packages needed
│
├── data/                 ← ⚠️ PUT YOUR CICIDS2017 CSV FILES HERE
│   └── (place CSV files here)
│
├── models/               ← Auto-created after training
│   ├── ids_lstm_model.h5      (trained model)
│   ├── label_encoder.pkl      (attack label mapping)
│   ├── scaler.pkl             (feature normalizer)
│   ├── feature_names.pkl      (feature list)
│   ├── X_train.npy            (training data)
│   ├── X_test.npy             (test data)
│   ├── training_history.png   (accuracy/loss chart)
│   └── confusion_matrix.png   (predictions heatmap)
│
├── templates/
│   ├── index.html        ← Dashboard home page
│   ├── analyze.html      ← Upload & analyze CSV page
│   └── metrics.html      ← Model performance page
│
└── static/
    └── css/
        └── style.css     ← All styling (cybersecurity theme)
```

---

## ⚙️ STEP 0 — PREREQUISITES

Before starting, make sure you have these installed on your computer:

### Check Python version
```
python --version
```
✅ You need Python 3.8, 3.9, or 3.10 (NOT 3.11+ for TensorFlow 2.13)

### Check pip
```
pip --version
```

---

## 📥 STEP 1 — DOWNLOAD THE CICIDS2017 DATASET

1. Go to this official website:
   **https://www.unb.ca/cic/datasets/ids-2017.html**

2. Scroll down and click **"Download"**

3. You will get a folder with these CSV files:
   ```
   Monday-WorkingHours.pcap_ISCX.csv
   Tuesday-WorkingHours.pcap_ISCX.csv
   Wednesday-workingHours.pcap_ISCX.csv
   Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   Friday-WorkingHours-Morning.pcap_ISCX.csv
   Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   ```

4. **Copy ALL these CSV files** into the `data/` folder of this project.

> 💡 TIP: If the dataset is too large (3.2 GB total), you can start with
> just 2-3 files like Wednesday + Friday to test the project.
> The model will still work.

---

## 📦 STEP 2 — INSTALL DEPENDENCIES

Open Terminal / Command Prompt inside the `IDS_Project/` folder.

### Create a virtual environment (RECOMMENDED)
```
python -m venv venv
```

### Activate virtual environment
On Windows:
```
venv\Scripts\activate
```
On Mac/Linux:
```
source venv/bin/activate
```

### Install all required packages
```
pip install -r requirements.txt
```

> ⏳ This will take 5-10 minutes. TensorFlow is a large package.

---

## 🔄 STEP 3 — PREPROCESS THE DATA

```
python preprocess.py
```

### What happens:
- Loads all CSV files from `data/` folder
- Cleans column names (CICIDS2017 has spaces in names)
- Removes infinity and NaN values
- Maps attack labels to clean names (e.g., "DoS Hulk" → "DoS")
- Selects top 20 most important features
- Normalizes all feature values
- Splits into training (80%) and test (20%) sets
- Saves everything to `models/` folder

### Expected output:
```
============================================================
   CICIDS2017 PREPROCESSING PIPELINE
============================================================
[INFO] Found 8 CSV files:
  Loading: Monday-WorkingHours.pcap_ISCX.csv
  ...
[INFO] Total records loaded: 2,830,743
[STEP 1] Cleaning column names...
[STEP 2] Mapping attack labels...
...
[INFO] Attack type distribution:
  BENIGN           :   2,271,320  (80.25%)
  DDoS             :     128,027  (4.52%)
  DoS              :      95,345  (3.37%)
  ...
============================================================
  PREPROCESSING COMPLETE!
  Train samples : 2,264,594
  Test samples  :   566,149
  Features used : 20
  Classes       : 11
============================================================

  Next step: Run  python train_model.py
```

---

## 🧠 STEP 4 — TRAIN THE LSTM MODEL

```
python train_model.py
```

### What happens:
- Loads preprocessed data from `models/` folder
- Builds a Bidirectional LSTM neural network
- Trains for up to 30 epochs (early stopping may stop earlier)
- Saves the best model automatically
- Generates accuracy/loss plots
- Prints classification report

### Expected output:
```
============================================================
   LSTM MODEL TRAINING - CICIDS2017 IDS
============================================================
[INFO] Loading preprocessed data...
[INFO] Building LSTM model...
Model: "sequential"
...
[INFO] Training for max 30 epochs...
Epoch 1/30 - loss: 0.2341 - accuracy: 0.9234
Epoch 2/30 - loss: 0.1123 - accuracy: 0.9567
...
[INFO] Evaluating on test set...
  Test Accuracy : 97.83%
  Test Loss     : 0.0812

[INFO] Classification Report:
                   precision  recall  f1-score
BENIGN               0.99      0.99     0.99
DDoS                 0.98      0.97     0.97
DoS                  0.96      0.95     0.95
...
============================================================
  TRAINING COMPLETE!
  Model saved : models/ids_lstm_model.h5
  Accuracy    : 97.83%
============================================================
```

> ⏳ Training time: 20-60 minutes depending on your computer.
> GPU will make it much faster.

---

## 🌐 STEP 5 — RUN THE WEB APPLICATION

```
python app.py
```

### Expected output:
```
==================================================
  IDS FLASK APP STARTING...
  Open browser at: http://127.0.0.1:5000
==================================================
```

### Open in browser:
```
http://127.0.0.1:5000
```

You will see the **NetGuard IDS Dashboard** with:
- 📊 Dashboard – Overview stats and charts
- 🔍 Analyze Traffic – Upload CSV to detect attacks
- 📈 Model Metrics – Architecture, confusion matrix, accuracy plots

---

## 🔍 STEP 6 — TEST THE DETECTION

### To test if detection works:

1. Go to **Analyze Traffic** page (sidebar)
2. Upload any CICIDS2017 CSV file (or use the original dataset files)
3. Click **"Run Detection"**
4. See results with:
   - Attack label for each network flow
   - Confidence percentage
   - Severity level (Normal / Medium / High / Critical)
   - Color-coded attack distribution chart

### What to upload for testing:
- Use any CSV from the CICIDS2017 dataset directly
- Or any CSV that has similar network flow features (same column names)

---

## ❓ COMMON ERRORS & FIXES

### Error: "No CSV files found in data/"
**Fix:** Make sure you placed CICIDS2017 CSV files inside the `data/` folder.

### Error: "Model not loaded"
**Fix:** You haven't trained the model yet. Run `python preprocess.py` then `python train_model.py`.

### Error: "ModuleNotFoundError: No module named tensorflow"
**Fix:** Run `pip install tensorflow==2.13.0` or re-run `pip install -r requirements.txt`

### Error: "CUDA out of memory"
**Fix:** Reduce BATCH_SIZE in `train_model.py` from 512 to 128.

### Error: "ValueError: operands could not be broadcast"
**Fix:** Your CSV might have different column names. Make sure it is a CICIDS2017 formatted CSV.

### Training is very slow
**Fix:** 
- Reduce epochs from 30 to 10 in `train_model.py`
- Use fewer CSV files (put only 2-3 CSVs in data/ folder)

---

## 📊 WHY CICIDS2017 IS BETTER THAN NSL-KDD

| Problem with NSL-KDD | How CICIDS2017 Fixes It |
|---------------------|------------------------|
| Created in 1999 – outdated attacks | Contains 2017 modern attacks |
| Detects normal traffic as suspicious | Real network captures reduce false positives |
| No web attacks (XSS, SQLi) | Includes complete web attack coverage |
| No botnet traffic | Contains botnet communication patterns |
| Synthetic dataset | Real network traffic from live systems |
| High false positive rate | Significantly lower false positive rate |

---
### What does this system do?
This system monitors network traffic data and uses a deep learning model to classify each network connection as either normal (BENIGN) or a specific type of attack.

### Why Bidirectional LSTM?
- **LSTM** = Long Short-Term Memory. It remembers patterns over time, making it ideal for sequential network traffic data.
- **Bidirectional** = reads the sequence both forward AND backward, capturing patterns that one direction might miss.

### What features does it use?
20 network flow features like:
- Flow Duration, Packet count, Byte count
- Inter-arrival times (IAT)
- TCP flag counts (SYN, RST, PSH)
- Window sizes, Segment sizes

### How accurate is it?
Expected accuracy: **95-98%** on CICIDS2017 test data.

---

## 📝 QUICK REFERENCE COMMANDS

```bash
# 1. Install packages
pip install -r requirements.txt

# 2. Preprocess data (run once)
python preprocess.py

# 3. Train model (run once, takes time)
python train_model.py

# 4. Start web app (run every time)
python app.py

# Open browser
http://127.0.0.1:5000
```

---

*Project: Real-Time Network Intrusion Detection System*
*Dataset: CICIDS2017 | Model: Bidirectional LSTM | Framework: TensorFlow + Flask*
