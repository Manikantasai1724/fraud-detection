# 🛡️ FraudShield — Bank Transaction Fraud Detector

AI-powered fraud detection using transaction, channel, location, and network-behavior signals.
Built with **XGBoost + SMOTE** + **Flask** + a dark-themed UI.

---

## 📁 Project Structure

```
fraud_app/
│
├── bank_transactions_data_2.csv   ← Your dataset
├── train_model.py                 ← Run this FIRST
├── app.py                         ← Flask backend
├── requirements.txt               ← Dependencies
├── README.md                      ← This file
│
├── model/                         ← Auto-created after training
│   ├── fraud_model.pkl
│   ├── feature_cols.pkl
│   ├── location_map.pkl
│   ├── occupation_map.pkl
│   ├── device_lookup.pkl
│   ├── ip_lookup.pkl
│   ├── high_amt_threshold.pkl
│   ├── low_bal_threshold.pkl
│   └── evaluation_metrics.pkl
│
└── templates/
    └── index.html                 ← Frontend UI
```

---

## 🚀 Setup & Run in VS Code

### Step 1 — Open Project in VS Code
```
File → Open Folder → Select the fraud_app folder
```

### Step 2 — Open Terminal in VS Code
```
Terminal → New Terminal   (or Ctrl + `)
```

### Step 3 — Create Virtual Environment
```bash
python -m venv venv
```

### Step 4 — Activate Virtual Environment
```bash
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 5 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 6 — Train the Model (FIRST TIME ONLY)
```bash
python train_model.py
```
✅ This creates all files inside the `/model/` folder.

### Step 7 — Start the Web App
```bash
python app.py
```

### Step 8 — Open in Browser
```
http://localhost:5000
```

---

## 🔍 How Fraud is Detected

This project currently uses **proxy (rule-based) labels** for training and evaluation.
It is useful for prototyping, but these are **not adjudicated bank fraud outcomes**.

| Signal | Threshold | Risk |
|--------|-----------|------|
| Login Attempts | ≥ 3 | 🔐 Account Takeover |
| Amount / Balance Ratio | > 80% | 💸 Balance Drain |
| Transaction Amount | > $878 (90th pct) | ⚠️ High Value |
| Transaction Duration | < 15 seconds | ⚡ Bot Activity |
| Device → Multiple Accounts | > 5 accounts | 📱 Fraud Ring |
| IP → Multiple Accounts | > 7 accounts | 🌐 Bot Network |

---

## 🧪 Test Transactions

**Fraudulent (should flag as FRAUD):**
- Amount: $1800, Balance: $2000, Login Attempts: 5, Duration: 8s
- Device: D000142, IP: 200.136.146.93 (both high-risk in dataset)

**Legitimate (should pass as LEGIT):**
- Amount: $120, Balance: $6500, Login Attempts: 1, Duration: 150s
- Any unique Device ID and IP

---

## 📊 Model Evaluation (Recall-First)

- Training uses a **temporal split** (train -> validation -> holdout test).
- Model pipeline uses **SMOTE oversampling** + **class-weighted XGBoost**.
- Hyperparameter search is tuned with **recall** as the scoring metric.
- Inference uses a **lower tuned decision threshold** (typically ~0.2 to 0.4) to improve fraud recall.
- Device/IP counts, encodings, and thresholds are fit on **train data only**.
- Training excludes direct proxy-label-defining fields from the final model feature subset to reduce target leakage.
- Metrics are saved in `model/evaluation_metrics.pkl` after running training.

Use this command to regenerate metrics:

```bash
python train_model.py
```

Then read the console output for:
- Validation: Precision, Recall, F1, PR-AUC, ROC-AUC
- Holdout Test: Precision, Recall, F1, PR-AUC, ROC-AUC

⚠️ Important: These metrics measure performance against proxy labels, not confirmed fraud investigation labels.

---

## 🔑 Key Feature Importances

Feature importance values are printed each time you run training:

```bash
python train_model.py
```

This project now intentionally drops leakage-prone proxy-defining features from the model input set
and includes additional transaction-cadence features, so importances will differ from earlier versions.
