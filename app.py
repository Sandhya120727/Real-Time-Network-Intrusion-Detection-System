"""
Flask IDS Application - app.py
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from pymongo import MongoClient
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from collections import Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)
app.secret_key = "ids_secret_2024"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

MODEL_FOLDER = "models/"

print("[INFO] Loading model and artifacts...")
try:
    from tensorflow.keras.models import load_model
    model    = load_model(os.path.join(MODEL_FOLDER, "ids_lstm_model.h5"), compile=False)
    le       = joblib.load(os.path.join(MODEL_FOLDER, "label_encoder.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_FOLDER, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_FOLDER, "feature_names.pkl"))
    print(f"[INFO] Model loaded. Classes: {list(le.classes_)}")
    MODEL_LOADED = True
except Exception as e:
    print(f"[WARNING] Model not loaded: {e}")
    MODEL_LOADED = False
    le = type('obj', (object,), {'classes_': []})()

ATTACK_LOG     = []
TOTAL_ANALYZED = 0
# MongoDB Atlas Connection
MONGO_URI = "mongodb+srv://ids_user:ids_password123@network-ids-cluster.px8vmbk.mongodb.net/?appName=network-ids-cluster"
client = MongoClient(MONGO_URI)
db = client["ids_database"]
attacks_collection = db["attack_logs"]
print("[INFO] MongoDB Atlas connected successfully!")

ATTACK_COLORS = {
    "BENIGN": "#4CAF50", "DDoS": "#F44336", "DoS": "#FF5722",
    "Web Attack": "#FF9800", "SQL Injection": "#9C27B0",
    "Port Scan": "#2196F3", "Botnet": "#795548",
    "FTP Brute Force": "#00BCD4", "SSH Brute Force": "#009688",
    "Heartbleed": "#E91E63", "Infiltration": "#607D8B",
}
SEVERITY_MAP = {
    "BENIGN": "Normal", "DDoS": "Critical", "DoS": "Critical",
    "Web Attack": "High", "SQL Injection": "Critical",
    "Port Scan": "Medium", "Botnet": "High",
    "FTP Brute Force": "Medium", "SSH Brute Force": "Medium",
    "Heartbleed": "Critical", "Infiltration": "High",
}

def predict_dataframe(df):
    available = [f for f in features if f in df.columns]
    if len(available) < 5:
        return None, "Not enough matching features. Upload a CICIDS2017 CSV."
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0.0
    X = df[features].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)
    X_lstm   = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    raw_preds  = model.predict(X_lstm, verbose=0)
    class_idx  = np.argmax(raw_preds, axis=1)
    confidence = np.max(raw_preds, axis=1)
    labels     = le.inverse_transform(class_idx)
    results = []
    for i, (label, conf) in enumerate(zip(labels, confidence)):
        results.append({
            "row": i + 1, "label": label,
            "confidence": round(float(conf) * 100, 2),
            "severity": SEVERITY_MAP.get(label, "Unknown"),
            "color": ATTACK_COLORS.get(label, "#607D8B"),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
    return results, None

@app.route("/")
def index():
    stats = {
        "total_analyzed": TOTAL_ANALYZED,
        "total_attacks": sum(1 for e in ATTACK_LOG if e["label"] != "BENIGN"),
        "model_loaded": MODEL_LOADED,
        "attack_types": len(le.classes_) if MODEL_LOADED else 0,
        "recent_log": ATTACK_LOG[-10:][::-1] if ATTACK_LOG else [],
    }
    return render_template("index.html", stats=stats)

@app.route("/analyze")
def analyze():
    return render_template("analyze.html", model_loaded=MODEL_LOADED)

@app.route("/predict", methods=["POST"])
def predict():
    global TOTAL_ANALYZED, ATTACK_LOG
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Run python train_model.py first."}), 400
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file"}), 400
    try:
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
        results, error = predict_dataframe(df)
        if error:
            return jsonify({"error": error}), 400
        TOTAL_ANALYZED += len(results)
        ATTACK_LOG.extend(results)
        if len(ATTACK_LOG) > 500:
            ATTACK_LOG = ATTACK_LOG[-500:]
        for r in results:
            attacks_collection.insert_one({
                "timestamp"  : datetime.now(),
                "ip_address" : request.remote_addr,
                "attack_type": r["label"],
                "confidence" : r["confidence"],
                "severity"   : r["severity"],
                "source"     : "csv_upload"
            })

        labels    = [r["label"] for r in results]
        dist      = dict(Counter(labels))
        n_attacks = sum(1 for l in labels if l != "BENIGN")
        return jsonify({
            "success": True, "total_rows": len(results),
            "attacks_found": n_attacks, "benign_count": len(results) - n_attacks,
            "distribution": dist, "results": results[:200],
        })
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
@app.route("/history")
def history():
    logs = list(attacks_collection.find().sort("timestamp", -1).limit(100))
    for log in logs:
        log["_id"] = str(log["_id"])
        log["timestamp"] = log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    return render_template("history.html", logs=logs)

@app.route("/metrics")
def metrics():
    chart_exists = os.path.exists(os.path.join(MODEL_FOLDER, "training_history.png"))
    cm_exists    = os.path.exists(os.path.join(MODEL_FOLDER, "confusion_matrix.png"))
    classes      = list(le.classes_) if MODEL_LOADED else []
    return render_template("metrics.html", model_loaded=MODEL_LOADED,
                           chart_exists=chart_exists, cm_exists=cm_exists, classes=classes)

@app.route("/api/stats")
def api_stats():
    if not ATTACK_LOG:
        return jsonify({"total": 0, "attacks": 0, "benign": 0, "distribution": {}})
    labels = [e["label"] for e in ATTACK_LOG]
    return jsonify({
        "total": TOTAL_ANALYZED,
        "attacks": sum(1 for l in labels if l != "BENIGN"),
        "benign": sum(1 for l in labels if l == "BENIGN"),
        "distribution": dict(Counter(labels)),
        "recent": ATTACK_LOG[-5:][::-1],
    })

@app.route("/model_image/<name>")
def model_image(name):
    allowed = {"training_history.png", "confusion_matrix.png"}
    if name not in allowed:
        return "Not found", 404
    return send_from_directory(MODEL_FOLDER, name)
@app.route("/live_capture", methods=["POST"])
def live_capture():
    global TOTAL_ANALYZED, ATTACK_LOG
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded."}), 400
    try:
        from capture import capture_live
        duration = int(request.json.get("duration", 30))
        output_file = capture_live(duration=duration)
        if not output_file:
            return jsonify({"error": "No packets captured. Make sure you have network activity."}), 400
        import pandas as pd
        df = pd.read_csv(output_file)
        df.replace([float('inf'), float('-inf')], 0, inplace=True)
        results, error = predict_dataframe(df)
        if error:
            return jsonify({"error": error}), 400
        TOTAL_ANALYZED += len(results)
        ATTACK_LOG.extend(results)
        labels = [r["label"] for r in results]
        from collections import Counter
        dist = dict(Counter(labels))
        n_attacks = sum(1 for l in labels if l != "BENIGN")
        return jsonify({
            "success": True,
            "total_rows": len(results),
            "attacks_found": n_attacks,
            "benign_count": len(results) - n_attacks,
            "distribution": dist,
            "results": results[:200],
        })
    except Exception as e:
        return jsonify({"error": f"Capture error: {str(e)}"}), 500
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  IDS FLASK APP STARTING...")
    print("  Open browser at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)