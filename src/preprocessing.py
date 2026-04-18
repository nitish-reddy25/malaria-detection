"""
preprocessing.py
----------------
Image loading, augmentation, and dataset splitting pipeline
for the Malaria Cell Detection project.

Dataset: NIH Malaria Cell Images Dataset
  - 13,779 Parasitized images
  - 13,779 Uninfected images
  Total: 27,558 images

Split: 80% train / 10% validation / 10% test (stratified)
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_PATH = "../data"          # Root folder containing Parasitized/ and Uninfected/
IMG_SIZE     = 128                # Resize all images to 128×128
BATCH_SIZE   = 32
RANDOM_STATE = 42

LABEL_MAP = {
    "Parasitized": 1,
    "Uninfected":  0,
}


# ── Core loading function ───────────────────────────────────────────────────────
def load_images_from_directory(directory: str, label: int):
    """
    Load all valid images from a directory, resize, and normalise.

    Args:
        directory: Path to folder containing images.
        label:     Integer class label (1 = Parasitized, 0 = Uninfected).

    Returns:
        images: List of float32 NumPy arrays of shape (IMG_SIZE, IMG_SIZE, 3).
        labels: List of integers.
    """
    images, labels = [], []
    skipped = 0

    for filename in sorted(os.listdir(directory)):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)

        if img is None:
            skipped += 1
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # BGR → RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))           # Resize
        img = img.astype(np.float32) / 255.0                 # Normalise [0, 1]

        images.append(img)
        labels.append(label)

    print(f"  Loaded {len(images)} images from '{os.path.basename(directory)}'"
          f" (skipped {skipped} unreadable files)")
    return images, labels


# ── Dataset builder ─────────────────────────────────────────────────────────────
def build_dataset(dataset_path: str = DATASET_PATH):
    """
    Load the full dataset and split into train / validation / test sets.

    Returns:
        X_train, X_val, X_test: NumPy arrays, shape (N, IMG_SIZE, IMG_SIZE, 3)
        y_train, y_val, y_test: NumPy arrays, shape (N,)
    """
    print("Loading dataset...")
    all_images, all_labels = [], []

    for class_name, label in LABEL_MAP.items():
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected directory not found: {class_dir}\n"
                f"Make sure your dataset is placed at '{dataset_path}/' with "
                f"subfolders 'Parasitized/' and 'Uninfected/'."
            )
        imgs, lbls = load_images_from_directory(class_dir, label)
        all_images.extend(imgs)
        all_labels.extend(lbls)

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"\nTotal images: {len(X)}")
    print(f"  Parasitized : {np.sum(y == 1)}")
    print(f"  Uninfected  : {np.sum(y == 0)}")

    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Second split: 50% of temp → validation, 50% → test  (10% / 10% overall)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"\nSplit sizes:")
    print(f"  Train      : {len(X_train)}")
    print(f"  Validation : {len(X_val)}")
    print(f"  Test       : {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Data augmentation ───────────────────────────────────────────────────────────
def get_data_generators(X_train, y_train, X_val, y_val):
    """
    Wrap training and validation data in Keras ImageDataGenerators.

    Training generator applies random augmentations to improve generalisation.
    Validation generator performs no augmentation (evaluation must be clean).

    Returns:
        train_gen, val_gen: Keras generator objects.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        brightness_range=[0.85, 1.15],
    )

    val_datagen = ImageDataGenerator()  # No augmentation for validation

    train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_gen   = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

    return train_gen, val_gen


# ── Single-array preprocessor (used by tests & predict.py) ─────────────────────
def preprocess_image_array(img_bgr: np.ndarray, expand_dims: bool = False) -> np.ndarray:
    """
    Preprocess a raw BGR NumPy image array (as returned by cv2.imread).

    Args:
        img_bgr:     uint8 array of shape (H, W, 3) in BGR colour order.
        expand_dims: If True, add batch dimension → shape (1, IMG_SIZE, IMG_SIZE, 3).

    Returns:
        float32 NumPy array, pixel values in [0.0, 1.0].
    """
    import cv2 as _cv2
    img = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB)
    img = _cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    if expand_dims:
        img = np.expand_dims(img, axis=0)
    return img


# ── Persistence helpers ─────────────────────────────────────────────────────────
def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, save_dir: str = "../data"):
    """Save preprocessed splits as .npy files for fast reloading."""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(save_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(save_dir, "y_test.npy"),  y_test)
    print(f"Splits saved to '{save_dir}/'")


def load_splits(save_dir: str = "../data"):
    """Load previously saved .npy split files."""
    X_train = np.load(os.path.join(save_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(save_dir, "X_val.npy"))
    X_test  = np.load(os.path.join(save_dir, "X_test.npy"))
    y_train = np.load(os.path.join(save_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(save_dir, "y_val.npy"))
    y_test  = np.load(os.path.join(save_dir, "y_test.npy"))
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    splits = build_dataset(DATASET_PATH)
    save_splits(*splits)
    print("\nPreprocessing complete.")
