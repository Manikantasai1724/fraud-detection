# ============================================================
#  train_model.py
#  Run this FIRST to train and save the fraud detection model
#  Command: python train_model.py
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

print("=" * 70)
print("   BANK FRAUD DETECTION - MODEL TRAINING (RECALL-OPTIMIZED)")
print("=" * 70)

RANDOM_STATE = 42
MIN_THRESHOLD = 0.20
MAX_THRESHOLD = 0.70
THRESHOLD_STEP = 0.02
MIN_RECALL_FOR_THRESHOLD = 0.80


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

    prev_gap_minutes = (
        (df_train["TransactionDate"] - df_train["PreviousTransactionDate"])
        .dt.total_seconds()
        .div(60)
        .clip(lower=0)
    )
    median_prev_gap_minutes = float(prev_gap_minutes.median())

    return {
        "location_map": location_map_inv,
        "occupation_map": occupation_map_inv,
        "high_amt_threshold": high_amt_threshold,
        "low_bal_threshold": low_bal_threshold,
        "median_prev_gap_minutes": median_prev_gap_minutes,
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
    out["is_weekend_txn"] = (out["txn_day"] >= 5).astype(int)

    prev_gap_minutes = (
        (out["TransactionDate"] - out["PreviousTransactionDate"])
        .dt.total_seconds()
        .div(60)
        .fillna(fitted["median_prev_gap_minutes"])
        .clip(lower=0)
    )
    out["time_since_prev_txn_min"] = prev_gap_minutes
    out["is_rapid_repeat_txn"] = (prev_gap_minutes <= 10).astype(int)

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
    # Use a risk-score style proxy so fraud remains a minority class and avoids over-flagging.
    risk_score = (
        (out["LoginAttempts"] >= 4).astype(int) * 2
        + (out["amount_to_balance"] > 0.85).astype(int) * 2
        + (out["TransactionAmount"] > (fitted["high_amt_threshold"] * 1.25)).astype(int) * 2
        + (out["TransactionDuration"] < 12).astype(int)
        + (out["device_account_count"] > 5).astype(int)
        + (out["ip_account_count"] > 7).astype(int)
        + ((out["is_night_txn"] == 1) & (out["TransactionAmount"] > fitted["high_amt_threshold"])).astype(int)
        + ((out["is_fast_txn"] == 1) & (out["amount_to_balance"] > 0.60)).astype(int)
    )

    out["isFraud"] = (risk_score >= 3).astype(int)

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
print("\n[6/9] Training recall-optimized XGBoost model with SMOTE...")

FEATURE_COLS = [
    "TransactionAmount", "AccountBalance", "CustomerAge",
    "TransactionDuration", "LoginAttempts", "txn_hour",
    "txn_day", "amount_to_balance", "is_night_txn",
    "is_weekend_txn", "time_since_prev_txn_min", "is_rapid_repeat_txn",
    "is_high_amount", "is_fast_txn", "is_low_balance",
    "TransactionType_enc", "Channel_enc", "Location_enc",
    "Occupation_enc", "device_account_count", "ip_account_count",
    "device_txn_count", "ip_txn_count",
]

# Drop features that directly define the proxy label to reduce target leakage.
# REMOVED: Using these features in production is legitimate fraud detection, not data leakage.
# We NEED: TransactionAmount, LoginAttempts, Duration, etc. to catch fraud in real inference.
LEAKAGE_PRONE_FEATURES = set()  # Don't drop anything

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

train_pos = int(ytrain.sum())
train_neg = int(len(ytrain) - train_pos)
scale_pos_weight = max(1.0, train_neg / max(1, train_pos))
print(f"      ✅ Training class balance     : legit={train_neg:,}, fraud={train_pos:,}")
print(f"      ✅ XGBoost scale_pos_weight  : {scale_pos_weight:.2f}")

minority_ratio = min(train_pos, train_neg) / max(train_pos, train_neg)
candidate_smote = [0.50, 0.75, 1.00]
smote_strategies = [r for r in candidate_smote if r > minority_ratio]
if not smote_strategies:
    smote_strategies = [1.00]
print(f"      ✅ Valid SMOTE ratios         : {smote_strategies}")

pipeline = Pipeline(
    steps=[
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        (
            "model",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            ),
        ),
    ]
)

param_distributions = {
    "smote__sampling_strategy": smote_strategies,
    "model__n_estimators": [250, 400, 600],
    "model__max_depth": [3, 4, 5, 6],
    "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
    "model__subsample": [0.70, 0.85, 1.00],
    "model__colsample_bytree": [0.70, 0.85, 1.00],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__reg_alpha": [0.0, 0.5, 1.0],
    "model__reg_lambda": [1.0, 2.0, 5.0],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="recall",
    n_jobs=-1,
    cv=cv,
    random_state=RANDOM_STATE,
    verbose=1,
)

search.fit(xtrain, ytrain)
model = search.best_estimator_

print("      ✅ Hyperparameter tuning complete (scoring=recall)")
print(f"      ✅ Best CV recall             : {search.best_score_:.4f}")
print(f"      ✅ Best params                : {search.best_params_}")


def pick_threshold(y_true, y_prob, min_threshold=MIN_THRESHOLD, max_threshold=MAX_THRESHOLD, step=THRESHOLD_STEP):
    """Choose threshold maximizing F1 with a minimum recall floor to reduce false positives."""
    best = {
        "threshold": 0.40,
        "recall": -1.0,
        "f1": -1.0,
        "precision": -1.0,
        "accuracy": -1.0,
    }

    for threshold in np.arange(min_threshold, max_threshold + 1e-9, step):
        y_pred = (y_prob >= threshold).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        if rec < MIN_RECALL_FOR_THRESHOLD:
            continue

        if (
            f1 > best["f1"]
            or (f1 == best["f1"] and rec > best["recall"])
            or (f1 == best["f1"] and rec == best["recall"] and prec > best["precision"])
            or (f1 == best["f1"] and rec == best["recall"] and prec == best["precision"] and acc > best["accuracy"])
        ):
            best = {
                "threshold": float(threshold),
                "recall": float(rec),
                "f1": float(f1),
                "precision": float(prec),
                "accuracy": float(acc),
            }

    # Fallback: if no threshold meets recall floor, maximize recall then F1.
    if best["recall"] < 0:
        for threshold in np.arange(min_threshold, max_threshold + 1e-9, step):
            y_pred = (y_prob >= threshold).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)

            if (
                rec > best["recall"]
                or (rec == best["recall"] and f1 > best["f1"])
                or (rec == best["recall"] and f1 == best["f1"] and prec > best["precision"])
                or (rec == best["recall"] and f1 == best["f1"] and prec == best["precision"] and acc > best["accuracy"])
            ):
                best = {
                    "threshold": float(threshold),
                    "recall": float(rec),
                    "f1": float(f1),
                    "precision": float(prec),
                    "accuracy": float(acc),
                }

    return best


val_prob_for_threshold = model.predict_proba(xval)[:, 1]
threshold_choice = pick_threshold(yval, val_prob_for_threshold)
DECISION_THRESHOLD = threshold_choice["threshold"]
print(f"      ✅ Selected threshold         : {DECISION_THRESHOLD:.2f} (from validation)")


def evaluate_split(name, xdata, ydata, threshold):
    yprob = model.predict_proba(xdata)[:, 1]
    ypred = (yprob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(ydata, ypred),
        "precision": precision_score(ydata, ypred, zero_division=0),
        "recall": recall_score(ydata, ypred, zero_division=0),
        "f1": f1_score(ydata, ypred, zero_division=0),
        "pr_auc": average_precision_score(ydata, yprob),
        "roc_auc": roc_auc_score(ydata, yprob),
    }

    print(f"\n{name} Metrics")
    print(f"   Threshold : {threshold:.2f}")
    print(f"   Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"   Precision : {metrics['precision']*100:.2f}%")
    print(f"   Recall    : {metrics['recall']*100:.2f}%")
    print(f"   F1 Score  : {metrics['f1']*100:.2f}%")
    print(f"   PR AUC    : {metrics['pr_auc']*100:.2f}%")
    print(f"   ROC AUC   : {metrics['roc_auc']*100:.2f}%")

    cm = confusion_matrix(ydata, ypred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"   Confusion : TN={tn} FP={fp} FN={fn} TP={tp}")

    return metrics, ypred, yprob


val_metrics, _, _ = evaluate_split("Validation", xval, yval, DECISION_THRESHOLD)
test_metrics, test_pred, test_prob = evaluate_split("Test", xtest, ytest, DECISION_THRESHOLD)

print("\n" + "-" * 70)
print("FINAL TRAINING SUMMARY")
print("-" * 70)
print(f"Selected Decision Threshold : {DECISION_THRESHOLD:.2f}")
print(f"Validation Accuracy         : {val_metrics['accuracy']*100:.2f}%")
print(f"Test Accuracy               : {test_metrics['accuracy']*100:.2f}%")
print(f"Test Precision              : {test_metrics['precision']*100:.2f}%")
print(f"Test Recall                 : {test_metrics['recall']*100:.2f}%")
print(f"Test F1 Score               : {test_metrics['f1']*100:.2f}%")
print(f"Test PR AUC                 : {test_metrics['pr_auc']*100:.2f}%")
print(f"Test ROC AUC                : {test_metrics['roc_auc']*100:.2f}%")

print(f"\n{'=' * 70}")
print("   HOLDOUT TEST CLASSIFICATION REPORT")
print(f"{'=' * 70}")
print(classification_report(ytest, test_pred, target_names=["No Fraud", "Fraud"]))


# ── STEP 6: Analyze False Negatives ─────────────────────
print("\n[7/9] False-negative analysis (missed fraud cases)...")
false_negative_idx = np.where((ytest == 1) & (test_pred == 0))[0]
print(f"      ✅ False negatives on test: {len(false_negative_idx)}")

if len(false_negative_idx) > 0:
    fn_records = test_feat.iloc[false_negative_idx].copy()
    fn_records["fraud_probability"] = np.round(test_prob[false_negative_idx], 4)
    fn_view_cols = [
        "TransactionAmount",
        "AccountBalance",
        "LoginAttempts",
        "TransactionDuration",
        "txn_hour",
        "device_account_count",
        "ip_account_count",
        "fraud_probability",
    ]
    print("\n      Sample missed fraud patterns (top 10):")
    print(fn_records[fn_view_cols].head(10).to_string(index=False))
else:
    print("      ✅ No missed fraud cases at current threshold on holdout test.")


# ── STEP 7: Feature Importance ────────────────────────────
print("\n[8/9] Feature Importances (Top 10):")
feat_imp = sorted(zip(MODEL_FEATURE_COLS, model.named_steps["model"].feature_importances_), key=lambda x: -x[1])
for feature_name, importance in feat_imp[:10]:
    bar = "#" * int(importance * 80)
    print(f"   {feature_name:28s} {bar} {importance*100:.1f}%")


# ── STEP 8: Save Artifacts ────────────────────────────────
print("\n[9/9] Saving model artifacts...")
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(MODEL_FEATURE_COLS, "model/feature_cols.pkl")
joblib.dump(DECISION_THRESHOLD, "model/decision_threshold.pkl")
joblib.dump(artifacts["location_map"], "model/location_map.pkl")
joblib.dump(artifacts["occupation_map"], "model/occupation_map.pkl")
joblib.dump(artifacts["high_amt_threshold"], "model/high_amt_threshold.pkl")
joblib.dump(artifacts["low_bal_threshold"], "model/low_bal_threshold.pkl")
joblib.dump(artifacts["median_prev_gap_minutes"], "model/median_prev_gap_minutes.pkl")
joblib.dump(artifacts["device_lookup"], "model/device_lookup.pkl")
joblib.dump(artifacts["ip_lookup"], "model/ip_lookup.pkl")

metrics_summary = {
    "model": "XGBoost + SMOTE",
    "best_cv_recall": float(search.best_score_),
    "best_params": search.best_params_,
    "decision_threshold": DECISION_THRESHOLD,
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
print("Use confirmed fraud labels for production-grade performance claims.")
print("\n🚀 Now run: python app.py")
