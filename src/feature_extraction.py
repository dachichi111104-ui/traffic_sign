import cv2
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

# Try importing skimage.feature.hog, fallback to OpenCV HOG
try:
    from skimage.feature import hog as skimage_hog
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    logger.info("scikit-image not installed. Falling back to OpenCV HOGDescriptor.")


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extracts HOG (Histogram of Oriented Gradients) features from a single RGB image.
    1. Converts image to Grayscale.
    2. Resizes image to (32, 32) if needed.
    3. Computes HOG feature vector.
    """
    if image.dtype in [np.float32, np.float64]:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    if gray.shape != (32, 32):
        gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

    if HAS_SKIMAGE:
        features = skimage_hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
    else:
        # OpenCV HOGDescriptor for 32x32 image with 8x8 cell and 16x16 block size
        hog_cv = cv2.HOGDescriptor(
            _winSize=(32, 32),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        features = hog_cv.compute(gray).flatten()

    return features


def extract_hog_batch(images: np.ndarray) -> np.ndarray:
    """
    Extracts HOG feature vectors for a dataset batch of images.
    Returns array of shape (N, feature_dim).
    """
    feature_list = []
    for img in images:
        feature_list.append(extract_hog_features(img))
    return np.array(feature_list, dtype=np.float32)
