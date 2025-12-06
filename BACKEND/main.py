from flask import Flask, request, jsonify
import numpy as np
import os
import pickle
import traceback
import logging
from collections import Counter

# Initialize App
app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- CONFIGURATION ---
MODEL_PATH = "model2.pkl"
SCALER_PATH = "scaler2.pkl"
LABEL_ENCODER_PATH = "label_encoder2.pkl"

# Strictly enforced feature order (14 Raw Features)
FEATURE_ORDER = [
    "Xl", "Yl", "Zl",
    "Flex1", "Flex2", "Flex3", "Flex5",
    "Xr", "Yr", "Zr",
    "Flex6", "Flex7", "Flex8", "Flex9"
]
EXPECTED_RAW_COUNT = len(FEATURE_ORDER)

# --- GLOBAL ASSETS ---
assets = {
    "model": None,
    "scaler": None,
    "encoder": None,
    "status": "unloaded",
    "error": None
}

def load_assets():
    """Loads ML artifacts if they aren't already loaded."""
    global assets
    if assets["status"] == "loaded": return True

    try:
        if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH]):
             raise FileNotFoundError("One or more model artifact files are missing.")

        with open(MODEL_PATH, "rb") as f: assets["model"] = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: assets["scaler"] = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f: assets["encoder"] = pickle.load(f)

        assets["status"] = "loaded"
        assets["error"] = None
        app.logger.info("Assets loaded successfully.")
        return True
    except Exception as e:
        assets["status"] = "error"
        assets["error"] = str(e)
        app.logger.error(f"Asset Load Error: {traceback.format_exc()}")
        return False

# --- FEATURE ENGINEERING ENGINE (V2: Now includes Median Filtering) ---
def engineer_features(raw_matrix):
    """
    Transforms Raw (N, 14) -> Filtered (N, 14) -> Engineered (N, 22).
    """
    X = raw_matrix
    epsilon = 1e-6

    # --- 1. Signal Smoothing (Median Filter - Window 3) ---
    # NOTE: This implementation is for batch prediction. For a true real-time filter (single row),
    # you'd need to send a history buffer (e.g., last 3 points) or modify the app to cache history.
    # We assume 'raw_matrix' is a batch of sequential data (N samples).

    # We use a simple 3-point median filter implementation in NumPy
    N = X.shape[0]
    X_filtered = np.zeros_like(X)

    # For each column (feature)
    for j in range(X.shape[1]):
        col = X[:, j]
        for i in range(N):
            # Define window bounds (ensuring bounds stay within the array)
            start = max(0, i - 1)
            end = min(N, i + 2)
            window = col[start:end]
            X_filtered[i, j] = np.median(window)

    X = X_filtered # Use filtered data for subsequent steps

    # --- 2. Feature Calculation ---

    # Magnitudes
    mag_l = np.sqrt(X[:,0]**2 + X[:,1]**2 + X[:,2]**2).reshape(-1, 1)
    mag_r = np.sqrt(X[:,7]**2 + X[:,8]**2 + X[:,9]**2).reshape(-1, 1)

    # Hand Distance
    dist = np.sqrt(
        (X[:,0]-X[:,7])**2 + (X[:,1]-X[:,8])**2 + (X[:,2]-X[:,9])**2
    ).reshape(-1, 1)

    # Flex Means
    mean_l = np.mean(X[:, [3,4,5,6]], axis=1).reshape(-1, 1)
    mean_r = np.mean(X[:, [10,11,12,13]], axis=1).reshape(-1, 1)

    # Ratios
    ratio_lr = mean_l / (mean_r + epsilon)
    ratio_idx_l = (X[:,3] / (X[:,4] + epsilon)).reshape(-1, 1)
    ratio_idx_r = (X[:,10] / (X[:,11] + epsilon)).reshape(-1, 1)

    # Concatenate: Original (14) + New (8) = 22 Features
    return np.hstack([
        X, mag_l, mag_r, dist, mean_l, mean_r, ratio_lr, ratio_idx_l, ratio_idx_r
    ])


# --- INPUT PARSING HELPER ---
def parse_input(payload):
    # ... (Keep existing parse_input function logic as is) ...
    raw_data = []

    if isinstance(payload, dict):
        # Case A: {"samples": [...]}
        if "samples" in payload:
            for s in payload["samples"]:
                raw_data.append([float(s.get(k, 0.0)) for k in FEATURE_ORDER])
        # Case B: Single sample dict
        elif all(k in payload for k in FEATURE_ORDER):
            raw_data.append([float(payload.get(k, 0.0)) for k in FEATURE_ORDER])
        else:
            raise ValueError("JSON keys do not match expected feature names or 'samples' format.")
    else:
        raise ValueError("Payload must be a JSON object (single sample or batch in 'samples').")

    if not raw_data:
        raise ValueError("No valid sensor data found in payload.")

    X_raw = np.array(raw_data)
    if X_raw.shape[1] != EXPECTED_RAW_COUNT:
        raise ValueError(f"Feature count mismatch. Expected {EXPECTED_RAW_COUNT}, got {X_raw.shape[1]}")

    return X_raw


# --- REST OF ROUTES ---
@app.route("/")
# ... (Keep existing home function) ...
def home():
    return jsonify({
        "service": "Gesture Recognition API",
        "status": "running",
        "model_status": assets["status"]
    })

@app.route("/health")
# ... (Keep existing health function) ...
def health():
    load_assets()
    return jsonify({
        "status": "ok",
        "assets_loaded": assets["status"],
        "last_error": assets["error"]
    })

@app.route("/predict", methods=["POST"])
def predict():
    # 0. Ensure assets are loaded
    if not load_assets():
        return jsonify({"status": "error", "message": "Model assets missing on server.", "detail": assets["error"]}), 503

    try:
        payload = request.get_json(force=True)

        # 1. Parse Input (N, 14)
        X_raw = parse_input(payload)

        # 2. Feature Engineering (N, 14) -> (N, 22) <-- Filtered data is used here
        X_engineered = engineer_features(X_raw)

        # 3. Scale Features
        X_scaled = assets["scaler"].transform(X_engineered)

        # 4. Predict
        probs = assets["model"].predict_proba(X_scaled)

        # 5. Decode Logic
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = assets["encoder"].inverse_transform(pred_indices)
        confidences = np.max(probs, axis=1)

        # 6. Majority Vote & Index Calculation
        if len(pred_labels) > 0:
            majority_label = Counter(pred_labels).most_common(1)[0][0]

            # Get the encoded integer index
            majority_index = int(assets["encoder"].transform([majority_label])[0])

            # Calculate average confidence of the majority class
            mask = (pred_labels == majority_label)
            final_confidence = float(np.mean(confidences[mask]))
        else:
            majority_label = "UNKNOWN"
            majority_index = -1
            final_confidence = 0.0

        return jsonify({
            "status": "success",
            "majority_gesture": majority_label,
            "majority_gesture_index": majority_index,
            "confidence": round(final_confidence, 4),
            "batch_details": {
                "all_predictions": pred_labels.tolist(),
                "all_confidences": confidences.tolist()
            },
            "batch_size": len(X_raw)
        })

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Prediction Crash: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": "Internal Server Error"}), 500

if __name__ == "__main__":
    # Load assets immediately on startup
    load_assets()
    app.run(host="0.0.0.0", port=8080)
