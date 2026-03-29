# 🛡️ FraudShield — Bank Transaction Fraud Detector

AI-powered fraud detection using **Device ID, IP Address, Location** + 17 other behavioral signals.
Built with **Random Forest ML** + **Flask** + a dark-themed UI.

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
│   └── low_bal_threshold.pkl
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

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| F1 Score | 100% |
| Recall | 100% |

---

## 🔑 Key Feature Importances

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Device Account Count | 31.3% |
| 2 | Device Transaction Count | 21.5% |
| 3 | IP Account Count | 11.7% |
| 4 | IP Transaction Count | 11.6% |
| 5 | Amount / Balance Ratio | 6.9% |
| 6 | Login Attempts | 5.3% |
