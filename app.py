 # ============================================================
#  app.py — Flask Backend for Fraud Detection
#  Run: python app.py
#  Then open: http://localhost:5000
# ============================================================

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# ── Load Model & All Saved Artifacts ──────────────────────
print("Loading fraud detection model...")

model              = joblib.load("model/fraud_model.pkl")
FEATURE_COLS       = joblib.load("model/feature_cols.pkl")
location_map       = joblib.load("model/location_map.pkl")
occupation_map     = joblib.load("model/occupation_map.pkl")
HIGH_AMT_THRESHOLD = joblib.load("model/high_amt_threshold.pkl")
LOW_BAL_THRESHOLD  = joblib.load("model/low_bal_threshold.pkl")
device_lookup      = joblib.load("model/device_lookup.pkl")
ip_lookup          = joblib.load("model/ip_lookup.pkl")

print(f"✅ Model loaded | Features: {len(FEATURE_COLS)}")
print(f"✅ Locations   : {len(location_map)} cities")
print(f"✅ Occupations : {len(occupation_map)} types")


# ── Helper: Compute Device/IP Risk Signals ─────────────────
def get_device_signals(device_id):
    """Look up how many accounts & transactions linked to this device."""
    acc_count = device_lookup.get("account_count", {}).get(device_id, 1)
    txn_count = device_lookup.get("txn_count",     {}).get(device_id, 1)
    return int(acc_count), int(txn_count)


def get_ip_signals(ip_address):
    """Look up how many accounts & transactions linked to this IP."""
    acc_count = ip_lookup.get("account_count", {}).get(ip_address, 1)
    txn_count = ip_lookup.get("txn_count",     {}).get(ip_address, 1)
    return int(acc_count), int(txn_count)


# ── Helper: Build Feature Vector ──────────────────────────
def build_features(form_data):
    """
    Extract and engineer all features from form input.
    Returns (feature_vector, feature_breakdown_dict)
    """
    # Raw inputs
    amount      = float(form_data["transaction_amount"])
    balance     = float(form_data["account_balance"])
    age         = 35  # Default age (customer info no longer collected)
    duration    = int(form_data["transaction_duration"])
    logins      = int(form_data["login_attempts"])
    txn_type    = form_data["transaction_type"]           # Debit / Credit
    channel     = form_data["channel"]                    # ATM / Online / Branch
    location    = form_data["location"]
    occupation  = "Professional"  # Default occupation (customer info no longer collected)
    device_id   = form_data["device_id"].strip()
    ip_address  = form_data["ip_address"].strip()

    # Time features (use current time since transaction_date no longer collected)
    dt = datetime.now()
    txn_hour     = dt.hour
    txn_day      = dt.weekday()

    # Financial derived features
    amount_to_balance = amount / (balance + 1)
    is_night_txn      = int((txn_hour >= 22) or (txn_hour <= 6))
    is_high_amount    = int(amount > HIGH_AMT_THRESHOLD)
    is_fast_txn       = int(duration < 15)
    is_low_balance    = int(balance < LOW_BAL_THRESHOLD)

    # Categorical encoding
    type_enc      = {"Debit": 1, "Credit": 2}.get(txn_type, 1)
    channel_enc   = {"ATM": 1, "Online": 2, "Branch": 3}.get(channel, 1)
    location_enc  = location_map.get(location, 0)
    occ_enc       = occupation_map.get(occupation, 0)

    # Device & IP network signals
    dev_acc_count, dev_txn_count = get_device_signals(device_id)
    ip_acc_count,  ip_txn_count  = get_ip_signals(ip_address)

    # Build feature vector in exact training order
    feature_vector = [
        amount,             balance,            age,
        duration,           logins,             txn_hour,
        txn_day,            amount_to_balance,  is_night_txn,
        is_high_amount,     is_fast_txn,        is_low_balance,
        type_enc,           channel_enc,        location_enc,
        occ_enc,            dev_acc_count,      ip_acc_count,
        dev_txn_count,      ip_txn_count
    ]

    # Breakdown for UI display
    breakdown = {
        "login_attempts"     : logins,
        "amount_to_balance"  : round(amount_to_balance * 100, 1),
        "is_night_txn"       : bool(is_night_txn),
        "is_high_amount"     : bool(is_high_amount),
        "is_fast_txn"        : bool(is_fast_txn),
        "device_accounts"    : dev_acc_count,
        "ip_accounts"        : ip_acc_count,
        "device_txn_count"   : dev_txn_count,
        "ip_txn_count"       : ip_txn_count,
        "txn_hour"           : txn_hour,
    }

    return np.array([feature_vector]), breakdown


# ════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template(
        "index.html",
        locations   = sorted(location_map.keys())
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts form data, runs fraud model, returns JSON result.
    """
    try:
        form_data = request.form.to_dict()

        required_fields = [
            "transaction_amount",
            "account_balance",
            "transaction_type",
            "channel",
            "transaction_duration",
            "device_id",
            "ip_address",
            "login_attempts",
            "location",
        ]
        missing_fields = [
            field for field in required_fields
            if field not in form_data or str(form_data[field]).strip() == ""
        ]
        if missing_fields:
            return jsonify({
                "success": False,
                "error": "Missing required inputs",
                "missing_fields": missing_fields,
            }), 400

        # Build features
        features, breakdown = build_features(form_data)

        # Run prediction
        prediction  = model.predict(features)[0]
        probas      = model.predict_proba(features)[0]
        fraud_prob  = float(probas[1])
        legit_prob  = float(probas[0])

        # Risk level
        if fraud_prob >= 0.75:
            risk_level = "HIGH"
            risk_color = "#ef4444"
        elif fraud_prob >= 0.45:
            risk_level = "MEDIUM"
            risk_color = "#f59e0b"
        else:
            risk_level = "LOW"
            risk_color = "#22c55e"

        # Active fraud signals
        signals = []
        if breakdown["login_attempts"] >= 3:
            signals.append({
                "icon"  : "🔐",
                "title" : "Multiple Login Attempts",
                "detail": f"{breakdown['login_attempts']} login attempts detected — possible account takeover"
            })
        if breakdown["amount_to_balance"] > 80:
            signals.append({
                "icon"  : "💸",
                "title" : "Balance Drain Detected",
                "detail": f"{breakdown['amount_to_balance']}% of account balance being transacted"
            })
        if breakdown["is_high_amount"]:
            signals.append({
                "icon"  : "⚠️",
                "title" : "Unusually High Amount",
                "detail": f"Transaction amount exceeds 90th percentile threshold (>${HIGH_AMT_THRESHOLD:.0f})"
            })
        if breakdown["is_fast_txn"]:
            signals.append({
                "icon"  : "⚡",
                "title" : "Suspiciously Fast Transaction",
                "detail": "Transaction completed in under 15 seconds — possible automated bot"
            })
        if breakdown["is_night_txn"]:
            signals.append({
                "icon"  : "🌙",
                "title" : "Night-Time Transaction",
                "detail": f"Transaction at {breakdown['txn_hour']}:00 hrs — outside normal hours"
            })
        if breakdown["device_accounts"] > 5:
            signals.append({
                "icon"  : "📱",
                "title" : "Device Linked to Multiple Accounts",
                "detail": f"This device is associated with {breakdown['device_accounts']} different accounts"
            })
        if breakdown["ip_accounts"] > 7:
            signals.append({
                "icon"  : "🌐",
                "title" : "IP Address Linked to Multiple Accounts",
                "detail": f"This IP address is linked to {breakdown['ip_accounts']} different accounts"
            })

        return jsonify({
            "success"        : True,
            "prediction"     : "FRAUD"      if prediction == 1 else "LEGITIMATE",
            "fraud_prob"     : round(fraud_prob * 100, 1),
            "legit_prob"     : round(legit_prob * 100, 1),
            "risk_level"     : risk_level,
            "risk_color"     : risk_color,
            "signals"        : signals,
            "breakdown"      : breakdown,
            "total_signals"  : len(signals)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "RandomForest", "features": len(FEATURE_COLS)})


# ── Run App ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   🛡️  FRAUD DETECTION SERVER STARTING")
    print("=" * 55)
    print(f"   URL     : http://localhost:5000")
    print(f"   Model   : Random Forest (100 trees)")
    print(f"   Features: {len(FEATURE_COLS)} (incl. Device/IP/Location)")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
