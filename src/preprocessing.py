import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Union, List, Optional
import logging

from src.config import (
    IMAGE_SIZE, RANDOM_STATE, NUM_CLASSES, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    PREPROCESSING_EXAMPLES_PATH, RESULT_DIR
)

logger = logging.getLogger(__name__)


def load_image(image_input: Union[str, Path, np.ndarray, bytes]) -> np.ndarray:
    """
    Reads an image from file path, bytes, or numpy array and converts to RGB format.
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Could not load image from path: {image_input}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes.")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = image_input.copy()
    else:
        raise TypeError("Input must be a file path, bytes, or numpy array.")
    
    return img_rgb


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Resizes image to target_size (width, height).
    """
    if image.shape[0] != target_size[1] or image.shape[1] != target_size[0]:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return image


def detect_and_crop_traffic_sign(image: Union[str, Path, np.ndarray, bytes]) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Automatically detects traffic sign region in a full scene photo using color thresholding (Red, Blue, Yellow)
    and geometric contour filtering (aspect ratio, solidity).
    Returns (cropped_rgb_image, bbox_tuple_or_None).
    """
    img_rgb = load_image(image)
    h, w = img_rgb.shape[:2]
    
    # If image is already small or roughly square icon size (< 120px), return as is
    if w <= 120 and h <= 120:
        return img_rgb, None

    # Convert to HSV color space
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # 1. Red Mask (Stop, Speed limits, Danger triangles, No Entry)
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 70, 50]), np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 2. Blue Mask (Mandatory / Direction signs)
    lower_blue, upper_blue = np.array([95, 80, 50]), np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 3. Yellow Mask (Warning signs)
    lower_yellow, upper_yellow = np.array([15, 80, 80]), np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine all sign color masks
    combined_mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_blue, mask_yellow))
    
    # Morphological cleaning to close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_bbox = None
    max_area = 0
    min_sign_area = (h * w) * 0.001  # At least 0.1% of image size
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_sign_area:
            continue
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = float(bw) / bh if bh > 0 else 0
        
        # Traffic signs are roughly square/circle/triangle/octagon (aspect ratio between 0.5 and 1.6)
        if 0.5 <= aspect_ratio <= 1.6:
            if area > max_area:
                max_area = area
                best_bbox = (x, y, bw, bh)
                
    if best_bbox is not None:
        bx, by, bw, bh = best_bbox
        # Add 15% padding around bounding box
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.15)
        
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)
        
        cropped = img_rgb[y1:y2, x1:x2]
        return cropped, (x1, y1, x2 - x1, y2 - y1)
    
    # If no candidate found, return original image
    return img_rgb, None


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on LAB color space (L channel).
    Enhances contrast under shadow, glare, or dark conditions without distorting colors.
    """
    if image.dtype != np.uint8:
        img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        img_uint8 = image

    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    if image.dtype != np.uint8 and image.max() <= 1.0:
        return img_clahe.astype(np.float32) / 255.0
    return img_clahe


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel values from [0, 255] to [0.0, 1.0] float32.
    """
    images_arr = np.asarray(images, dtype=np.float32)
    if images_arr.max() > 1.0:
        images_arr /= 255.0
    return images_arr


def prepare_image_pipeline(image_input: Union[str, Path, np.ndarray, bytes], target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Complete inference preprocessing pipeline for a single image:
    Input -> RGB -> Resize (32x32) -> CLAHE -> Normalize -> (1, 32, 32, 3)
    """
    img_rgb = load_image(image_input)
    img_resized = resize_image(img_rgb, target_size=target_size)
    img_clahe = apply_clahe(img_resized)
    img_norm = normalize_pixels(img_clahe)
    return np.expand_dims(img_norm, axis=0)


def split_data(
    images: List[np.ndarray],
    labels: np.ndarray,
    test_ratio: float = TEST_RATIO,
    val_ratio: float = VAL_RATIO,
    random_state: int = RANDOM_STATE
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data into Train (70%), Validation (15%), and Test (15%) sets without data leakage.
    Splitting occurs BEFORE any data augmentation.
    """
    # 1. Separate Test Set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_ratio, random_state=random_state, stratify=labels
    )
    
    # 2. Separate Validation Set (15% / (70% + 15%) = 0.1765 of remaining temp data)
    val_adjusted_ratio = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adjusted_ratio, random_state=random_state, stratify=y_temp
    )

    logger.info(
        f"Data split complete -> Train: {len(X_train)} (70%), Val: {len(X_val)} (15%), Test: {len(X_test)} (15%)"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def augment_single_image(image: np.ndarray) -> np.ndarray:
    """
    Applies realistic random data augmentation FOR TRAINING DATA ONLY:
    - Rotation (-12 to +12 degrees)
    - Translation (shift)
    - Brightness/contrast scaling (simulates day/night/glare)
    - Mild Gaussian blur / noise (simulates rain/fog/blur)
    NO horizontal flip (preserves directional traffic sign meanings).
    """
    h, w = image.shape[:2]
    
    # Random rotation
    angle = np.random.uniform(-12, 12)
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    aug = cv2.warpAffine(image, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    # Random shift
    tx = np.random.randint(-3, 4)
    ty = np.random.randint(-3, 4)
    M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
    aug = cv2.warpAffine(aug, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Random brightness/contrast adjustment
    brightness_factor = np.random.uniform(0.8, 1.2)
    aug = np.clip(aug.astype(np.float32) * brightness_factor, 0, 255 if aug.max() > 1.0 else 1.0)

    # Weather simulation: 30% chance of mild blur/noise
    if np.random.rand() < 0.3:
        aug = cv2.GaussianBlur(aug.astype(np.uint8), (3, 3), 0)
    
    return aug.astype(image.dtype)


def save_preprocessing_visualization(sample_image: np.ndarray, save_path: Path = PREPROCESSING_EXAMPLES_PATH) -> Path:
    """
    Generates and saves visual comparison of Original Image -> Resized -> CLAHE -> Normalized -> Augmented.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img_rgb = load_image(sample_image)
    img_resized = resize_image(img_rgb, (32, 32))
    img_clahe = apply_clahe(img_resized)
    img_norm = normalize_pixels(img_clahe)
    img_aug = augment_single_image(img_resized)

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))

    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Original Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_resized)
    axes[1].set_title("2. Resized (32x32)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(img_clahe)
    axes[2].set_title("3. CLAHE Contrast", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(img_norm)
    axes[3].set_title("4. Normalized [0, 1]", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(img_aug)
    axes[4].set_title("5. Augmented (Train Only)", fontsize=10, fontweight="bold")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved preprocessing visualization to {save_path}")
    return save_path
