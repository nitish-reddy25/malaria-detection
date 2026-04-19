import sys, os, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from preprocessing import preprocess_image_array, IMG_SIZE
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="preprocessing not importable")

@pytest.fixture
def img():
    return np.random.default_rng(42).integers(0, 256, (200, 200, 3), dtype=np.uint8)

def test_shape(img):
    assert preprocess_image_array(img).shape == (IMG_SIZE, IMG_SIZE, 3)

def test_dtype(img):
    assert preprocess_image_array(img).dtype == np.float32

def test_range(img):
    r = preprocess_image_array(img)
    assert r.min() >= 0.0 and r.max() <= 1.0

def test_batch_dim(img):
    assert preprocess_image_array(img, expand_dims=True).shape == (1, IMG_SIZE, IMG_SIZE, 3)

def test_black():
    r = preprocess_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    assert np.all(r == 0.0)

def test_white():
    np.testing.assert_allclose(
        preprocess_image_array(np.full((100, 100, 3), 255, dtype=np.uint8)), 1.0, atol=1e-6)
