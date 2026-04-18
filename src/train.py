"""
train.py
--------
Training pipeline for all malaria detection models.

Usage:
    python src/train.py --model cnn     --epochs 50 --batch_size 32
    python src/train.py --model vgg19   --epochs 30 --batch_size 32
    python src/train.py --model resnet50 --epochs 30 --batch_size 32
    python src/train.py --model hybrid  --epochs 50 --batch_size 32
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

from models import get_model
from preprocessing import build_dataset, get_data_generators, load_splits, save_splits

# ── Paths ───────────────────────────────────────────────────────────────────────
RESULTS_DIR = "../results"
MODELS_DIR  = "../results/saved_models"
LOGS_DIR    = "../results/training_logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ── Training function ────────────────────────────────────────────────────────────
def train(model_name: str, epochs: int = 50, batch_size: int = 32,
          learning_rate: float = 1e-3, use_cached: bool = True):
    """
    Load data, build model, train with callbacks, save checkpoint.

    Args:
        model_name:    One of 'cnn', 'vgg19', 'resnet50', 'hybrid'.
        epochs:        Maximum training epochs.
        batch_size:    Mini-batch size.
        learning_rate: Initial Adam learning rate.
        use_cached:    If True and .npy splits exist, skip raw loading.

    Returns:
        history: Keras History object.
        model:   Trained tf.keras.Model.
    """
    print(f"\n{'='*60}")
    print(f"  Training model: {model_name.upper()}")
    print(f"  Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {learning_rate}")
    print(f"{'='*60}\n")

    # ── Load data ──
    cache_path = "../data"
    cache_files = [f"{cache_path}/{f}.npy" for f in
                   ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]]

    if use_cached and all(os.path.exists(f) for f in cache_files):
        print("Loading cached .npy splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = load_splits(cache_path)
    else:
        print("Building dataset from raw images...")
        X_train, X_val, X_test, y_train, y_val, y_test = build_dataset()
        save_splits(X_train, X_val, X_test, y_train, y_val, y_test, cache_path)

    train_gen, val_gen = get_data_generators(X_train, y_train, X_val, y_val)

    # ── Build model ──
    model = get_model(model_name)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    # ── Callbacks ──
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
    log_path        = os.path.join(LOGS_DIR,   f"{model_name}_training_log.csv")

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(log_path, append=False),
    ]

    # ── Fit ──
    steps_per_epoch  = len(X_train) // batch_size
    validation_steps = len(X_val)   // batch_size

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nModel checkpoint saved to: {checkpoint_path}")
    print(f"Training log saved to:     {log_path}")

    return history, model, X_test, y_test


# ── CLI ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a malaria detection model.")
    parser.add_argument(
        "--model", type=str, default="cnn",
        choices=["cnn", "vgg19", "resnet50", "hybrid"],
        help="Model architecture to train."
    )
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Rebuild dataset from raw images even if .npy caches exist."
    )
    args = parser.parse_args()

    history, model, X_test, y_test = train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_cached=not args.no_cache,
    )

    # Quick test-set evaluation after training
    print("\nEvaluating on test set...")
    from evaluate import evaluate_model
    evaluate_model(model, X_test, y_test, model_name=args.model)
