"""
tests/test_preprocessing.py
----------------------------
Unit tests for the preprocessing pipeline.
Run with: pytest tests/
"""

import sys
import os
import numpy as np
import pytest

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import preprocess_image_array, IMG_SIZE


# ── Fixture: a fake 200×200 RGB image ───────────────────────────────────────────
@pytest.fixture
def fake_image():
    """A random uint8 BGR image mimicking what cv2.imread would return."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)


# ── Tests ────────────────────────────────────────────────────────────────────────
def test_output_shape(fake_image):
    """Preprocessed image must have shape (IMG_SIZE, IMG_SIZE, 3)."""
    result = preprocess_image_array(fake_image)
    assert result.shape == (IMG_SIZE, IMG_SIZE, 3), (
        f"Expected shape ({IMG_SIZE}, {IMG_SIZE}, 3), got {result.shape}"
    )


def test_output_dtype(fake_image):
    """Preprocessed image must be float32."""
    result = preprocess_image_array(fake_image)
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


def test_normalisation_range(fake_image):
    """All pixel values must be in [0.0, 1.0] after normalisation."""
    result = preprocess_image_array(fake_image)
    assert result.min() >= 0.0, "Minimum pixel value below 0.0"
    assert result.max() <= 1.0, "Maximum pixel value above 1.0"


def test_batch_expansion(fake_image):
    """When expand_dims=True (default for inference), shape is (1, H, W, 3)."""
    from preprocessing import preprocess_image_array
    result = preprocess_image_array(fake_image, expand_dims=True)
    assert result.shape == (1, IMG_SIZE, IMG_SIZE, 3)


def test_all_black_image():
    """An all-black image should normalise to all zeros without errors."""
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    result = preprocess_image_array(black)
    assert np.all(result == 0.0)


def test_all_white_image():
    """An all-white image should normalise to all ones without errors."""
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    result = preprocess_image_array(white)
    np.testing.assert_allclose(result, 1.0, atol=1e-6)
