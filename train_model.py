# ============================================================
#  train_model.py
#  Run this FIRST to train and save the fraud detection model
#  Command: python train_model.py
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score

print("=" * 60)
print("   BANK FRAUD DETECTION — MODEL TRAINING")
print("=" * 60)

# ── STEP 1: Load Dataset ───────────────────────────────────
print("\n[1/7] Loading dataset...")
data = pd.read_csv("bank_transactions_data_2.csv")
print(f"      ✅ Loaded {len(data):,} rows × {len(data.columns)} columns")

# ── STEP 2: Parse Dates & Time Features ───────────────────
print("\n[2/7] Extracting time features...")
data["TransactionDate"]         = pd.to_datetime(data["TransactionDate"])
data["PreviousTransactionDate"] = pd.to_datetime(data["PreviousTransactionDate"])
data["txn_hour"]  = data["TransactionDate"].dt.hour
data["txn_day"]   = data["TransactionDate"].dt.dayofweek
data["txn_month"] = data["TransactionDate"].dt.month

# ── STEP 3: Device / IP / Location Network Features ───────
print("\n[3/7] Engineering Device, IP & Location features...")

# How many unique accounts used the same device? (device sharing = fraud ring)
device_account_count        = data.groupby("DeviceID")["AccountID"].nunique()
data["device_account_count"]= data["DeviceID"].map(device_account_count)

# How many transactions from same device? (velocity signal)
device_txn_count            = data.groupby("DeviceID")["TransactionAmount"].count()
data["device_txn_count"]    = data["DeviceID"].map(device_txn_count)

# How many unique accounts from same IP? (IP sharing = fraud network)
ip_account_count            = data.groupby("IP Address")["AccountID"].nunique()
data["ip_account_count"]    = data["IP Address"].map(ip_account_count)

# How many transactions from same IP? (IP velocity)
ip_txn_count                = data.groupby("IP Address")["TransactionAmount"].count()
data["ip_txn_count"]        = data["IP Address"].map(ip_txn_count)

print(f"      ✅ Device sharing max: {data['device_account_count'].max()} accounts/device")
print(f"      ✅ IP sharing max    : {data['ip_account_count'].max()} accounts/IP")

# ── STEP 4: Financial Features ────────────────────────────
print("\n[4/7] Engineering financial features...")

data["amount_to_balance"] = data["TransactionAmount"] / (data["AccountBalance"] + 1)

HIGH_AMT_THRESHOLD        = data["TransactionAmount"].quantile(0.90)
LOW_BAL_THRESHOLD         = data["AccountBalance"].quantile(0.15)

data["is_night_txn"]      = ((data["txn_hour"] >= 22) | (data["txn_hour"] <= 6)).astype(int)
data["is_high_amount"]    = (data["TransactionAmount"] > HIGH_AMT_THRESHOLD).astype(int)
data["is_fast_txn"]       = (data["TransactionDuration"] < 15).astype(int)
data["is_low_balance"]    = (data["AccountBalance"] < LOW_BAL_THRESHOLD).astype(int)

# ── STEP 5: Create Fraud Labels ───────────────────────────
print("\n[5/7] Creating fraud labels using real-world signals...")

data["isFraud"] = (
    (data["LoginAttempts"] >= 3)           |   # Brute force / account takeover
    (data["amount_to_balance"] > 0.80)     |   # Balance draining attack
    (data["TransactionAmount"] > 1500)     |   # Unusually large transaction
    (data["TransactionDuration"] < 15)     |   # Bot / automated script
    (data["device_account_count"] > 5)     |   # Device used by fraud ring
    (data["ip_account_count"] > 7)             # IP linked to multiple accounts
).astype(int)

fraud_count = data["isFraud"].sum()
print(f"      ✅ Legitimate : {len(data) - fraud_count:,}")
print(f"      🚨 Fraudulent : {fraud_count:,}")
print(f"      📊 Fraud Rate : {data['isFraud'].mean()*100:.1f}%")

# ── STEP 6: Encode Categorical Columns ────────────────────
print("\n[6/7] Encoding categorical features...")

data["TransactionType_enc"] = data["TransactionType"].map({"Debit": 1, "Credit": 2})
data["Channel_enc"]         = data["Channel"].map({"ATM": 1, "Online": 2, "Branch": 3})
data["Location_enc"]        = data["Location"].astype("category").cat.codes
data["Occupation_enc"]      = data["CustomerOccupation"].astype("category").cat.codes

# Save all encoding maps for use in Flask app
location_map    = dict(enumerate(data["Location"].astype("category").cat.categories))
occupation_map  = dict(enumerate(data["CustomerOccupation"].astype("category").cat.categories))
location_map_inv    = {v: k for k, v in location_map.items()}
occupation_map_inv  = {v: k for k, v in occupation_map.items()}

# ── STEP 7: Train Model ───────────────────────────────────
print("\n[7/7] Training Random Forest model...")

FEATURE_COLS = [
    "TransactionAmount",    "AccountBalance",       "CustomerAge",
    "TransactionDuration",  "LoginAttempts",        "txn_hour",
    "txn_day",              "amount_to_balance",    "is_night_txn",
    "is_high_amount",       "is_fast_txn",          "is_low_balance",
    "TransactionType_enc",  "Channel_enc",          "Location_enc",
    "Occupation_enc",       "device_account_count", "ip_account_count",
    "device_txn_count",     "ip_txn_count"
]

x = data[FEATURE_COLS].values
y = data["isFraud"].values

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators = 100,
    class_weight = "balanced",
    random_state = 42,
    n_jobs       = -1
)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

print(f"\n{'=' * 60}")
print(f"   TRAINING COMPLETE — MODEL PERFORMANCE")
print(f"{'=' * 60}")
print(f"   Accuracy  : {model.score(xtest, ytest)*100:.2f}%")
print(f"   F1 Score  : {f1_score(ytest, ypred)*100:.2f}%")
print(f"   Recall    : {recall_score(ytest, ypred)*100:.2f}%")
print(f"\n{classification_report(ytest, ypred, target_names=['No Fraud','Fraud'])}")

print("Feature Importances (Top 10):")
feat_imp = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
for f, imp in feat_imp[:10]:
    bar = "█" * int(imp * 100)
    print(f"   {f:28s} {bar} {imp*100:.1f}%")

# ── Save Model & Metadata ─────────────────────────────────
os.makedirs("model", exist_ok=True)

joblib.dump(model,              "model/fraud_model.pkl")
joblib.dump(FEATURE_COLS,       "model/feature_cols.pkl")
joblib.dump(location_map_inv,   "model/location_map.pkl")
joblib.dump(occupation_map_inv, "model/occupation_map.pkl")
joblib.dump(HIGH_AMT_THRESHOLD, "model/high_amt_threshold.pkl")
joblib.dump(LOW_BAL_THRESHOLD,  "model/low_bal_threshold.pkl")

# Save device/IP lookup tables for real-time scoring
device_lookup = data.groupby("DeviceID").agg(
    account_count=("AccountID", "nunique"),
    txn_count    =("TransactionAmount", "count")
).to_dict()
ip_lookup = data.groupby("IP Address").agg(
    account_count=("AccountID", "nunique"),
    txn_count    =("TransactionAmount", "count")
).to_dict()

joblib.dump(device_lookup, "model/device_lookup.pkl")
joblib.dump(ip_lookup,     "model/ip_lookup.pkl")

print(f"\n✅ All model files saved to /model/ folder:")
for f in os.listdir("model"):
    size = os.path.getsize(f"model/{f}")
    print(f"   → {f:40s} ({size/1024:.1f} KB)")

print("\n🚀 Now run: python app.py")
