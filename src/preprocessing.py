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


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Applies Non-linear Gamma Correction to adjust global luminance/contrast.
    Gamma > 1.0 brightens shadow areas; Gamma < 1.0 dims overexposed areas.
    """
    if image.dtype != np.uint8:
        img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        img_uint8 = image.copy()
        
    inv_gamma = 1.0 / max(gamma, 1e-5)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_corrected = cv2.LUT(img_uint8, table)
    
    if image.dtype != np.uint8 and image.max() <= 1.0:
        return img_corrected.astype(np.float32) / 255.0
    return img_corrected


def apply_unsharp_mask(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 1.0, amount: float = 1.5, threshold: int = 0) -> np.ndarray:
    """
    Applies Unsharp Masking edge sharpening.
    Enhances fine details like numbers (e.g., speed limits), arrows, and symbols on traffic signs.
    """
    if image.dtype != np.uint8:
        img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        img_uint8 = image.copy()

    blurred = cv2.GaussianBlur(img_uint8, kernel_size, sigma)
    sharpened = cv2.addWeighted(img_uint8, 1.0 + amount, blurred, -amount, 0)
    
    if threshold > 0:
        low_contrast_mask = np.abs(img_uint8.astype(np.int16) - blurred.astype(np.int16)) < threshold
        np.copyto(sharpened, img_uint8, where=low_contrast_mask)
        
    if image.dtype != np.uint8 and image.max() <= 1.0:
        return np.clip(sharpened.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_perspective_transform(image: np.ndarray, max_warp: float = 0.10) -> np.ndarray:
    """
    Applies random perspective transform augmentation.
    Simulates camera viewing angles and tilted roadside perspective distortions.
    """
    h, w = image.shape[:2]
    dx = w * max_warp
    dy = h * max_warp
    
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [np.random.uniform(-dx, dx), np.random.uniform(-dy, dy)],
        [w + np.random.uniform(-dx, dx), np.random.uniform(-dy, dy)],
        [np.random.uniform(-dx, dx), h + np.random.uniform(-dy, dy)],
        [w + np.random.uniform(-dx, dx), h + np.random.uniform(-dy, dy)]
    ])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return warped


def apply_cutout(image: np.ndarray, n_holes: int = 1, length: int = 8) -> np.ndarray:
    """
    Applies Cutout / Random Erasing augmentation.
    Fills random square region(s) with zero/mean values to force CNN to rely on full sign context.
    """
    h, w = image.shape[:2]
    img_cut = image.copy()
    
    for _ in range(n_holes):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        
        img_cut[y1:y2, x1:x2] = 0
        
    return img_cut


def apply_motion_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Applies directional motion blur convolution.
    Simulates camera motion blur when captured from a moving vehicle.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    if np.random.rand() > 0.5:
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    else:
        np.fill_diagonal(kernel, 1)
    kernel /= kernel_size
    
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel values from [0, 255] to [0.0, 1.0] float32.
    """
    images_arr = np.asarray(images, dtype=np.float32)
    if images_arr.max() > 1.0:
        images_arr /= 255.0
    return images_arr


def normalize_zscore(images: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies Per-Channel Z-score Normalization (X - Mean) / (Std + 1e-7).
    Returns (normalized_images, train_mean, train_std).
    Calculates mean and std from Training set only to prevent data leakage.
    """
    images_arr = np.asarray(images, dtype=np.float32)
    if mean is None:
        mean = np.mean(images_arr, axis=(0, 1, 2), keepdims=True)
    if std is None:
        std = np.std(images_arr, axis=(0, 1, 2), keepdims=True)
        
    normalized = (images_arr - mean) / (std + 1e-7)
    return normalized, mean, std


def apply_imbalanced_sampling(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "none",
    target_samples_per_class: Optional[int] = None,
    random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies class balancing techniques for imbalanced datasets (e.g. GTSRB):
    - 'random_undersample': Randomly samples majority classes down to target count.
    - 'nearmiss': Keeps majority class samples closest to minority class samples using K-NN logic.
    - 'cluster_centroids': Replaces majority class samples with KMeans cluster centroids.
    - 'none': Returns original X, y unmodified.
    """
    if method == "none" or method is None:
        return X, y
        
    np.random.seed(random_state)
    classes, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts)) if target_samples_per_class is None else target_samples_per_class
    
    X_resampled = []
    y_resampled = []
    
    is_image = (X.ndim == 4)
    orig_shape = X.shape
    if is_image:
        X_flat = X.reshape(len(X), -1)
    else:
        X_flat = X
        
    if method == "random_undersample":
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) > min_count:
                selected_idx = np.random.choice(idx, size=min_count, replace=False)
            else:
                selected_idx = idx
            X_resampled.append(X[selected_idx])
            y_resampled.append(y[selected_idx])
        return np.concatenate(X_resampled, axis=0), np.concatenate(y_resampled, axis=0)
        
    elif method == "nearmiss":
        from sklearn.neighbors import NearestNeighbors
        minority_class = classes[np.argmin(counts)]
        minority_idx = np.where(y == minority_class)[0]
        minority_samples = X_flat[minority_idx]
        
        nn = NearestNeighbors(n_neighbors=min(5, len(minority_samples)))
        nn.fit(minority_samples)
        
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) > min_count:
                distances, _ = nn.kneighbors(X_flat[idx])
                mean_dist = np.mean(distances, axis=1)
                selected_sub_idx = np.argsort(mean_dist)[:min_count]
                selected_idx = idx[selected_sub_idx]
            else:
                selected_idx = idx
            X_resampled.append(X[selected_idx])
            y_resampled.append(y[selected_idx])
        return np.concatenate(X_resampled, axis=0), np.concatenate(y_resampled, axis=0)
        
    elif method == "cluster_centroids":
        from sklearn.cluster import KMeans
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) > min_count:
                kmeans = KMeans(n_clusters=min_count, random_state=random_state, n_init=3)
                kmeans.fit(X_flat[idx])
                centroids = kmeans.cluster_centers_
                if is_image:
                    centroids = centroids.reshape((min_count,) + orig_shape[1:])
                X_resampled.append(centroids.astype(X.dtype))
                y_resampled.append(np.full(min_count, c))
            else:
                X_resampled.append(X[idx])
                y_resampled.append(y[idx])
        return np.concatenate(X_resampled, axis=0), np.concatenate(y_resampled, axis=0)
        
    else:
        logger.warning(f"Unknown sampling method '{method}'. Returning original dataset.")
        return X, y


def preprocess_single_image(
    image_input: Union[str, Path, np.ndarray, bytes],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    use_clahe: bool = True,
    use_gamma: bool = False,
    use_unsharp: bool = False,
    norm_type: str = "minmax",
    mean: Optional[Union[np.ndarray, List[float]]] = None,
    std: Optional[Union[np.ndarray, List[float]]] = None,
    gamma_val: float = 1.2
) -> np.ndarray:
    """
    Unified sequential preprocessing pipeline for a single image:
    1. Read / Convert RGB (load_image)
    2. Resize to 32x32 (resize_image)
    3. Optional CLAHE (apply_clahe)
    4. Optional Gamma Correction (apply_gamma_correction)
    5. Optional Unsharp Masking (apply_unsharp_mask)
    6. Normalization (minmax /255 OR per-channel Z-score with train mean/std)
    """
    # 1. Load RGB
    img = load_image(image_input)
    # 2. Resize
    img = resize_image(img, target_size=target_size)
    # 3. CLAHE
    if use_clahe:
        img = apply_clahe(img)
    # 4. Gamma Correction
    if use_gamma:
        img = apply_gamma_correction(img, gamma=gamma_val)
    # 5. Unsharp Mask
    if use_unsharp:
        img = apply_unsharp_mask(img)
    # 6. Normalization
    img_float = img.astype(np.float32) if img.dtype != np.float32 else img.copy()
    if norm_type == "zscore":
        if mean is None or std is None:
            m = np.mean(img_float, axis=(0, 1), keepdims=True)
            s = np.std(img_float, axis=(0, 1), keepdims=True)
        else:
            m = np.array(mean, dtype=np.float32)
            s = np.array(std, dtype=np.float32)
            if m.ndim == 1:
                m = m.reshape(1, 1, -1)
            if s.ndim == 1:
                s = s.reshape(1, 1, -1)
        img_norm = (img_float - m) / (s + 1e-7)
    else:
        if img_float.max() > 1.0:
            img_norm = img_float / 255.0
        else:
            img_norm = img_float

    return img_norm


def preprocess_dataset_batch(
    images: List[np.ndarray],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    use_clahe: bool = True,
    use_gamma: bool = False,
    use_unsharp: bool = False,
    norm_type: str = "minmax",
    mean: Optional[Union[np.ndarray, List[float]]] = None,
    std: Optional[Union[np.ndarray, List[float]]] = None
) -> Tuple[np.ndarray, Optional[List[float]], Optional[List[float]]]:
    """
    Applies the unified sequential preprocessing pipeline to a dataset batch.
    If norm_type == 'zscore' and mean is None:
        Calculates mean and std from this batch (Train set) and returns them as python lists.
    Returns: (processed_images_array, train_mean_list, train_std_list)
    """
    processed_list = []
    for img in images:
        p_img = load_image(img)
        p_img = resize_image(p_img, target_size=target_size)
        if use_clahe:
            p_img = apply_clahe(p_img)
        if use_gamma:
            p_img = apply_gamma_correction(p_img)
        if use_unsharp:
            p_img = apply_unsharp_mask(p_img)
        processed_list.append(p_img.astype(np.float32))

    batch_arr = np.array(processed_list, dtype=np.float32)

    calc_mean = None
    calc_std = None

    if norm_type == "zscore":
        if mean is None or std is None:
            c_mean = np.mean(batch_arr, axis=(0, 1, 2), keepdims=True)
            c_std = np.std(batch_arr, axis=(0, 1, 2), keepdims=True)
            calc_mean = c_mean.flatten().tolist()
            calc_std = c_std.flatten().tolist()
            m = c_mean
            s = c_std
        else:
            m = np.array(mean, dtype=np.float32).reshape(1, 1, 1, -1)
            s = np.array(std, dtype=np.float32).reshape(1, 1, 1, -1)
            calc_mean = list(mean) if isinstance(mean, (list, tuple)) else mean.flatten().tolist()
            calc_std = list(std) if isinstance(std, (list, tuple)) else std.flatten().tolist()
        
        batch_norm = (batch_arr - m) / (s + 1e-7)
    else:
        if batch_arr.max() > 1.0:
            batch_norm = batch_arr / 255.0
        else:
            batch_norm = batch_arr

    return batch_norm, calc_mean, calc_std


def prepare_image_pipeline(
    image_input: Union[str, Path, np.ndarray, bytes],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    preprocessing_config: Optional[dict] = None
) -> np.ndarray:
    """
    Complete inference preprocessing pipeline for a single image.
    Uses metadata preprocessing_config if provided.
    """
    if preprocessing_config is None:
        preprocessing_config = {
            "use_clahe": True,
            "use_gamma": False,
            "use_unsharp": False,
            "norm_type": "minmax",
            "train_mean": None,
            "train_std": None
        }

    img_norm = preprocess_single_image(
        image_input,
        target_size=target_size,
        use_clahe=preprocessing_config.get("use_clahe", True),
        use_gamma=preprocessing_config.get("use_gamma", False),
        use_unsharp=preprocessing_config.get("use_unsharp", False),
        norm_type=preprocessing_config.get("norm_type", "minmax"),
        mean=preprocessing_config.get("train_mean"),
        std=preprocessing_config.get("train_std")
    )
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
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_ratio, random_state=random_state, stratify=labels
    )
    
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
    - Perspective transform (camera tilt simulation)
    - Rotation (-12 to +12 degrees)
    - Translation (shift ±3 px)
    - Brightness/contrast scaling (simulates day/night/glare)
    - Motion blur or Gaussian blur (weather & speed blur simulation)
    - Cutout / Random erasing (occlusion simulation)
    NO horizontal flip (preserves directional traffic sign meanings).
    """
    h, w = image.shape[:2]
    aug = image.copy()
    
    # 1. Perspective Transform (25% chance)
    if np.random.rand() < 0.25:
        aug = apply_perspective_transform(aug, max_warp=0.10)

    # 2. Random rotation (-12 to +12 degrees)
    angle = np.random.uniform(-12, 12)
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    # 3. Random shift
    tx = np.random.randint(-3, 4)
    ty = np.random.randint(-3, 4)
    M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
    aug = cv2.warpAffine(aug, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 4. Random brightness/contrast adjustment
    brightness_factor = np.random.uniform(0.8, 1.2)
    aug = np.clip(aug.astype(np.float32) * brightness_factor, 0, 255 if aug.max() > 1.0 else 1.0)

    # 5. Motion blur or Gaussian blur (30% chance)
    if np.random.rand() < 0.3:
        if np.random.rand() < 0.5:
            aug = cv2.GaussianBlur(aug.astype(np.uint8), (3, 3), 0)
        else:
            aug = apply_motion_blur(aug.astype(np.uint8), kernel_size=3)

    # 6. Cutout / Random Erasing (20% chance)
    if np.random.rand() < 0.20:
        aug = apply_cutout(aug, n_holes=1, length=6)
    
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
