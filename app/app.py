"""
app.py
------
Flask web application for the Malaria Cell Detection system.

Routes:
  GET  /           → Upload page
  POST /predict    → Run inference, return result page
  GET  /api/health → JSON health check
"""

import os
import time
import uuid
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

# ── Configuration ───────────────────────────────────────────────────────────────
IMG_SIZE       = 128
THRESHOLD      = 0.50
UPLOAD_FOLDER  = os.path.join("static", "uploads")
ALLOWED_EXTS   = {"png", "jpg", "jpeg", "bmp", "tiff"}
MAX_FILE_MB    = 10
MODELS_DIR     = os.path.join("..", "results", "saved_models")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]    = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024  # bytes

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# ── Lazy model cache ─────────────────────────────────────────────────────────────
_model_cache: dict = {}


def get_model(model_name: str) -> tf.keras.Model:
    """Load a model checkpoint, caching it in memory after first load."""
    if model_name not in _model_cache:
        path = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        logging.info(f"Loading model '{model_name}' from {path}")
        _model_cache[model_name] = tf.keras.models.load_model(path)
    return _model_cache[model_name]


# ── Helpers ──────────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def preprocess_image(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Cannot read the uploaded image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


# ── Routes ───────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate file upload ──
    if "file" not in request.files:
        return render_template("index.html", error="No file selected.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTS).upper()}"
        )

    # ── Save upload ──
    filename  = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # ── Inference ──
    model_name = request.form.get("model", "hybrid")
    start      = time.time()

    try:
        model      = get_model(model_name)
        img_tensor = preprocess_image(save_path)
        prob       = float(model.predict(img_tensor, verbose=0)[0][0])
    except FileNotFoundError as e:
        return render_template("index.html", error=str(e))
    except Exception as e:
        logging.error(f"Inference error: {e}")
        return render_template("index.html", error="Inference failed. Please try again.")

    elapsed = round(time.time() - start, 3)

    # ── Interpret result ──
    pred_class   = int(prob >= THRESHOLD)
    label        = "Parasitized" if pred_class == 1 else "Uninfected"
    confidence   = prob if pred_class == 1 else (1.0 - prob)
    needs_review = confidence < 0.80

    image_url = url_for("static", filename=f"uploads/{filename}")

    logging.info(
        f"Prediction | model={model_name} label={label} "
        f"confidence={confidence:.3f} time={elapsed}s"
    )

    return render_template(
        "result.html",
        image_url    = image_url,
        prediction   = label,
        confidence   = f"{confidence * 100:.1f}",
        model_used   = model_name.upper(),
        time_taken   = elapsed,
        needs_review = needs_review,
    )


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models_available": list(_model_cache.keys())})


# ── Entry point ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
