from flask import Flask, request, jsonify
import numpy as np, os, pickle, traceback, logging
from collections import Counter
from tensorflow.keras.models import load_model

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Config ---
MODEL_PATH = "best_model_test.h5"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
FEATURE_ORDER = [
    "Xl","Yl","Zl",
    "Flex1","Flex2","Flex3","Flex5",
    "Xr","Yr","Zr",
    "Flex6","Flex7","Flex8","Flex9","Flex10"
]
MAX_SAMPLES = 2000  # guardrail

# --- Lazy-loaded globals ---
model = None
scaler = None
label_encoder = None
assets_error = None
latest_prediction = {"gesture": None, "prob": None}

def load_assets():
    """Load model/scaler/encoder once, on demand."""
    global model, scaler, label_encoder, assets_error
    if model is not None and scaler is not None and label_encoder is not None:
        return True
    try:
        app.logger.info("Loading assets...")
        m = load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            s = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        model, scaler, label_encoder = m, s, le
        assets_error = None
        app.logger.info("Assets loaded.")
        return True
    except Exception as e:
        assets_error = f"{type(e).__name__}: {e}"
        app.logger.error("Asset load failed: %s\n%s", e, traceback.format_exc())
        return False

def parse_batch_features(payload):
    """
    Accepts any of:
      {"features": [[...15...],[...15...]]}
      {"features": [ {dict row}, {dict row} ]}
      {"samples":  [ {dict row}, {dict row} ]}
      single dict row with keys (treated as one sample)
    Returns np.ndarray (N,15)
    """
    # samples: list of dicts
    if isinstance(payload, dict) and "samples" in payload:
        rows = payload["samples"]
        X = [[float(r[k]) for k in FEATURE_ORDER] for r in rows]
        return np.asarray(X, dtype=float)

    # features: list of lists OR list of dicts
    if isinstance(payload, dict) and "features" in payload:
        rows = payload["features"]
        if not rows: raise ValueError("'features' must be non-empty")
        first = rows[0]
        if isinstance(first, (list, tuple)):
            arr = np.asarray(rows, dtype=float)
            return arr.reshape(1, -1) if arr.ndim == 1 else arr
        if isinstance(first, dict):
            X = [[float(r[k]) for k in FEATURE_ORDER] for r in rows]
            return np.asarray(X, dtype=float)

    # single dict row
    if isinstance(payload, dict):
        row = [float(payload[k]) for k in FEATURE_ORDER]
        return np.asarray([row], dtype=float)

    raise ValueError(
        "Invalid feature format. Provide one of: "
        '{"samples":[{row},...]}, {"features":[[...],[...]]}, '
        '{"features":[{row},{row}]}, or a single row with keys: '
        + ", ".join(FEATURE_ORDER)
    )

@app.route("/")
def home():
    return "Glove Prediction API running"

@app.route("/health")
def health():
    files_exist = {p: os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH]}
    loaded = dict(model=model is not None, scaler=scaler is not None, label_encoder=label_encoder is not None)
    return jsonify({"status": "ok", "files_exist": files_exist, "loaded": loaded, "assets_error": assets_error})

@app.route("/diag")
def diag():
    listing = []
    for f in sorted(os.listdir(".")):
        if os.path.isfile(f):
            try:
                listing.append({"name": f, "size": os.path.getsize(f)})
            except Exception:
                listing.append({"name": f, "size": None})
    return jsonify({"cwd": os.getcwd(), "files": listing})

@app.route("/predict", methods=["POST"])
def predict():
    global latest_prediction
    try:
        if not load_assets():
            return jsonify({"status": "error", "message": "Assets failed to load", "details": assets_error}), 503

        payload = request.get_json(force=True)
        X = parse_batch_features(payload)  # (N,15)

        if X.ndim != 2 or X.shape[1] != len(FEATURE_ORDER):
            raise ValueError(f"Expected shape (N,{len(FEATURE_ORDER)}), got {X.shape}")
        if X.shape[0] > MAX_SAMPLES:
            raise ValueError(f"Too many samples (> {MAX_SAMPLES}).")

        Xs = scaler.transform(X)
        probs = model.predict(Xs)  # (N, n_classes)
        pred_idxs = np.argmax(probs, axis=1)
        labels = label_encoder.inverse_transform(pred_idxs)

        # Majority vote
        majority_label, majority_count = Counter(labels).most_common(1)[0]
        mask = (labels == majority_label)
        conf = float(probs[mask].max()) if mask.any() else float(probs.max())

        latest_prediction = {"gesture": majority_label, "prob": conf}

        return jsonify({
            "status": "ok",
            "majority_gesture": majority_label,
            "majority_count": int(majority_count),
            "total_samples": int(X.shape[0]),
            "confidence_hint": conf,
            "all_predictions": labels.tolist(),
            "all_max_probs": probs.max(axis=1).tolist()
        })

    except Exception as e:
        app.logger.error("Predict error: %s\n%s", e, traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 400

latest_prediction = {"gesture": None, "prob": None}

@app.route("/get-prediction", methods=["GET"])
def get_prediction():
    return jsonify(latest_prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
