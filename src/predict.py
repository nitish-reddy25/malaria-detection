"""
predict.py
----------
Single-image inference utility.

Usage:
    python src/predict.py --image path/to/cell_image.png --model cnn
    python src/predict.py --image path/to/cell_image.png --model hybrid
"""

import argparse
import os
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE       = 128
MODELS_DIR     = "../results/saved_models"
THRESHOLD      = 0.50
CLASS_NAMES    = {0: "Uninfected", 1: "Parasitized"}
CLASS_EMOJI    = {0: "🟢", 1: "🔴"}


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load a single image from disk and prepare it for model inference.

    Returns:
        Array of shape (1, IMG_SIZE, IMG_SIZE, 3), dtype float32.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)   # shape: (1, 128, 128, 3)


def load_model(model_name: str) -> tf.keras.Model:
    """Load a saved model checkpoint by name."""
    checkpoint = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No saved model found at: {checkpoint}\n"
            f"Train the model first with: python src/train.py --model {model_name}"
        )
    model = tf.keras.models.load_model(checkpoint)
    return model


def predict(image_path: str, model_name: str = "hybrid", threshold: float = THRESHOLD) -> dict:
    """
    Run inference on a single image.

    Returns:
        result: dict with keys:
            - label        (str)  : 'Parasitized' or 'Uninfected'
            - probability  (float): raw sigmoid output [0, 1]
            - confidence   (float): confidence in the predicted class
            - needs_review (bool) : True if confidence < 0.80
    """
    img_tensor = preprocess_image(image_path)
    model      = load_model(model_name)

    prob         = float(model.predict(img_tensor, verbose=0)[0][0])
    pred_class   = int(prob >= threshold)
    confidence   = prob if pred_class == 1 else (1.0 - prob)
    needs_review = confidence < 0.80

    result = {
        "label":        CLASS_NAMES[pred_class],
        "probability":  round(prob, 4),
        "confidence":   round(confidence, 4),
        "needs_review": needs_review,
    }

    # Pretty-print
    emoji = CLASS_EMOJI[pred_class]
    print(f"\n{'─'*40}")
    print(f"  Image : {os.path.basename(image_path)}")
    print(f"  Model : {model_name.upper()}")
    print(f"  Result: {emoji} {result['label']}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    if needs_review:
        print(f"  ⚠️  Low confidence — manual review recommended.")
    print(f"{'─'*40}\n")

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict malaria infection from a cell image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument(
        "--model", type=str, default="hybrid",
        choices=["cnn", "vgg19", "resnet50", "hybrid"],
        help="Which trained model to use."
    )
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Classification threshold (default: 0.50).")
    args = parser.parse_args()

    predict(args.image, model_name=args.model, threshold=args.threshold)
