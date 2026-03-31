# ============================================================
#  train_model.py
#  Run this FIRST to train and save the fraud detection model
#  Command: python train_model.py
# ============================================================

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

print("=" * 70)
print("   BANK FRAUD DETECTION - MODEL TRAINING (LEAKAGE-REDUCED)")
print("=" * 70)

RANDOM_STATE = 42


# ── STEP 1: Load Dataset ───────────────────────────────────
print("\n[1/8] Loading dataset...")
data = pd.read_csv("bank_transactions_data_2.csv")
print(f"      ✅ Loaded {len(data):,} rows × {len(data.columns)} columns")

# ── STEP 2: Parse Dates & Temporal Split ──────────────────
print("\n[2/8] Parsing dates and creating temporal splits...")
data["TransactionDate"] = pd.to_datetime(data["TransactionDate"])
data["PreviousTransactionDate"] = pd.to_datetime(data["PreviousTransactionDate"])
data = data.sort_values("TransactionDate").reset_index(drop=True)


def temporal_split(df, train_ratio=0.70, val_ratio=0.15):
    """Split by time order to mimic forward-looking production behavior."""
    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


train_df, val_df, test_df = temporal_split(data)
print(f"      ✅ Time split -> train: {len(train_df):,}, val: {len(val_df):,}, test: {len(test_df):,}")


# ── STEP 3: Fit Train-Only Artifacts ──────────────────────
print("\n[3/8] Fitting thresholds, encoders, and network lookups on train only...")


def fit_artifacts(df_train):
    """Fit all thresholds/encodings/lookups using training data only."""
    location_categories = sorted(df_train["Location"].astype(str).unique())
    occupation_categories = sorted(df_train["CustomerOccupation"].astype(str).unique())

    location_map_inv = {name: idx for idx, name in enumerate(location_categories)}
    occupation_map_inv = {name: idx for idx, name in enumerate(occupation_categories)}

    high_amt_threshold = df_train["TransactionAmount"].quantile(0.90)
    low_bal_threshold = df_train["AccountBalance"].quantile(0.15)

    device_lookup_fit = df_train.groupby("DeviceID").agg(
        account_count=("AccountID", "nunique"),
        txn_count=("TransactionAmount", "count"),
    ).to_dict()

    ip_lookup_fit = df_train.groupby("IP Address").agg(
        account_count=("AccountID", "nunique"),
        txn_count=("TransactionAmount", "count"),
    ).to_dict()

    return {
        "location_map": location_map_inv,
        "occupation_map": occupation_map_inv,
        "high_amt_threshold": high_amt_threshold,
        "low_bal_threshold": low_bal_threshold,
        "device_lookup": device_lookup_fit,
        "ip_lookup": ip_lookup_fit,
    }


artifacts = fit_artifacts(train_df)


# ── STEP 4: Build Features ────────────────────────────────
print("\n[4/8] Building feature sets with train-fitted artifacts...")


def build_features(df, fitted):
    """Create model features using only train-fitted artifacts."""
    out = df.copy()

    out["txn_hour"] = out["TransactionDate"].dt.hour
    out["txn_day"] = out["TransactionDate"].dt.dayofweek

    out["amount_to_balance"] = out["TransactionAmount"] / (out["AccountBalance"] + 1)
    out["is_night_txn"] = ((out["txn_hour"] >= 22) | (out["txn_hour"] <= 6)).astype(int)
    out["is_high_amount"] = (out["TransactionAmount"] > fitted["high_amt_threshold"]).astype(int)
    out["is_fast_txn"] = (out["TransactionDuration"] < 15).astype(int)
    out["is_low_balance"] = (out["AccountBalance"] < fitted["low_bal_threshold"]).astype(int)

    out["TransactionType_enc"] = out["TransactionType"].map({"Debit": 1, "Credit": 2}).fillna(0).astype(int)
    out["Channel_enc"] = out["Channel"].map({"ATM": 1, "Online": 2, "Branch": 3}).fillna(0).astype(int)
    out["Location_enc"] = out["Location"].map(fitted["location_map"]).fillna(0).astype(int)
    out["Occupation_enc"] = out["CustomerOccupation"].map(fitted["occupation_map"]).fillna(0).astype(int)

    out["device_account_count"] = out["DeviceID"].map(fitted["device_lookup"].get("account_count", {})).fillna(1).astype(int)
    out["device_txn_count"] = out["DeviceID"].map(fitted["device_lookup"].get("txn_count", {})).fillna(1).astype(int)
    out["ip_account_count"] = out["IP Address"].map(fitted["ip_lookup"].get("account_count", {})).fillna(1).astype(int)
    out["ip_txn_count"] = out["IP Address"].map(fitted["ip_lookup"].get("txn_count", {})).fillna(1).astype(int)

    # Proxy labels for bootstrapping only; replace with adjudicated labels in production.
    out["isFraud"] = (
        (out["LoginAttempts"] >= 3)
        | (out["amount_to_balance"] > 0.80)
        | (out["TransactionAmount"] > 1500)
        | (out["TransactionDuration"] < 15)
        | (out["device_account_count"] > 5)
        | (out["ip_account_count"] > 7)
    ).astype(int)

    return out


train_feat = build_features(train_df, artifacts)
val_feat = build_features(val_df, artifacts)
test_feat = build_features(test_df, artifacts)

print(f"      ✅ Device sharing max (train): {train_feat['device_account_count'].max()} accounts/device")
print(f"      ✅ IP sharing max (train)    : {train_feat['ip_account_count'].max()} accounts/IP")


def print_label_stats(name, df):
    fraud_count = int(df["isFraud"].sum())
    print(f"      {name:<5} -> Legit: {len(df) - fraud_count:,} | Fraud: {fraud_count:,} | Rate: {df['isFraud'].mean()*100:.1f}%")


print("\n[5/8] Proxy-label distribution...")
print_label_stats("Train", train_feat)
print_label_stats("Val", val_feat)
print_label_stats("Test", test_feat)


# ── STEP 5: Train Model ───────────────────────────────────
print("\n[6/8] Training Random Forest model...")

FEATURE_COLS = [
    "TransactionAmount", "AccountBalance", "CustomerAge",
    "TransactionDuration", "LoginAttempts", "txn_hour",
    "txn_day", "amount_to_balance", "is_night_txn",
    "is_high_amount", "is_fast_txn", "is_low_balance",
    "TransactionType_enc", "Channel_enc", "Location_enc",
    "Occupation_enc", "device_account_count", "ip_account_count",
    "device_txn_count", "ip_txn_count",
]

# Drop features that directly define the proxy label to reduce target leakage.
LEAKAGE_PRONE_FEATURES = {
    "TransactionAmount",
    "AccountBalance",
    "TransactionDuration",
    "LoginAttempts",
    "amount_to_balance",
    "is_high_amount",
    "is_fast_txn",
    "device_account_count",
    "ip_account_count",
}

MODEL_FEATURE_COLS = [col for col in FEATURE_COLS if col not in LEAKAGE_PRONE_FEATURES]

print(f"      ✅ Total engineered features : {len(FEATURE_COLS)}")
print(f"      ✅ Model features in use     : {len(MODEL_FEATURE_COLS)}")
print(f"      ⚠️  Dropped leakage-prone    : {', '.join(sorted(LEAKAGE_PRONE_FEATURES))}")

xtrain = train_feat[MODEL_FEATURE_COLS].values
ytrain = train_feat["isFraud"].values
xval = val_feat[MODEL_FEATURE_COLS].values
yval = val_feat["isFraud"].values
xtest = test_feat[MODEL_FEATURE_COLS].values
ytest = test_feat["isFraud"].values

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(xtrain, ytrain)
print("      ✅ Model fit complete")


def evaluate_split(name, xdata, ydata):
    ypred = model.predict(xdata)
    yprob = model.predict_proba(xdata)[:, 1]

    metrics = {
        "accuracy": accuracy_score(ydata, ypred),
        "precision": precision_score(ydata, ypred, zero_division=0),
        "recall": recall_score(ydata, ypred, zero_division=0),
        "f1": f1_score(ydata, ypred, zero_division=0),
        "pr_auc": average_precision_score(ydata, yprob),
        "roc_auc": roc_auc_score(ydata, yprob),
    }

    print(f"\n{name} Metrics")
    print(f"   Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"   Precision : {metrics['precision']*100:.2f}%")
    print(f"   Recall    : {metrics['recall']*100:.2f}%")
    print(f"   F1 Score  : {metrics['f1']*100:.2f}%")
    print(f"   PR AUC    : {metrics['pr_auc']*100:.2f}%")
    print(f"   ROC AUC   : {metrics['roc_auc']*100:.2f}%")

    return metrics, ypred


val_metrics, _ = evaluate_split("Validation", xval, yval)
test_metrics, test_pred = evaluate_split("Test", xtest, ytest)

print(f"\n{'=' * 70}")
print("   HOLDOUT TEST CLASSIFICATION REPORT")
print(f"{'=' * 70}")
print(classification_report(ytest, test_pred, target_names=["No Fraud", "Fraud"]))


# ── STEP 6: Feature Importance ────────────────────────────
print("\n[7/8] Feature Importances (Top 10):")
feat_imp = sorted(zip(MODEL_FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
for feature_name, importance in feat_imp[:10]:
    bar = "#" * int(importance * 80)
    print(f"   {feature_name:28s} {bar} {importance*100:.1f}%")


# ── STEP 7: Save Artifacts ────────────────────────────────
print("\n[8/8] Saving model artifacts...")
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(MODEL_FEATURE_COLS, "model/feature_cols.pkl")
joblib.dump(artifacts["location_map"], "model/location_map.pkl")
joblib.dump(artifacts["occupation_map"], "model/occupation_map.pkl")
joblib.dump(artifacts["high_amt_threshold"], "model/high_amt_threshold.pkl")
joblib.dump(artifacts["low_bal_threshold"], "model/low_bal_threshold.pkl")
joblib.dump(artifacts["device_lookup"], "model/device_lookup.pkl")
joblib.dump(artifacts["ip_lookup"], "model/ip_lookup.pkl")

metrics_summary = {
    "validation": val_metrics,
    "test": test_metrics,
    "notes": "Metrics are measured against proxy rule-based labels, not adjudicated fraud outcomes.",
}
joblib.dump(metrics_summary, "model/evaluation_metrics.pkl")

print("\n✅ All model files saved to /model/ folder:")
for model_file in sorted(os.listdir("model")):
    size = os.path.getsize(f"model/{model_file}")
    print(f"   -> {model_file:40s} ({size/1024:.1f} KB)")

print("\n⚠️  NOTE: Metrics are against proxy labels generated from heuristics.")
print("   Use confirmed fraud labels for production-grade performance claims.")
print("\n🚀 Now run: python app.py")
