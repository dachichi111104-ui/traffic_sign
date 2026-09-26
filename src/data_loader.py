import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging

from src.config import (
    RAW_DATA_DIR, GTSRB_RAW_DIR, SAMPLE_DATA_DIR, EXTERNAL_TEST_DIR,
    IMAGE_SIZE, NUM_CLASSES, GTSRB_CLASSES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_dataset_dir(base_path: Path = RAW_DATA_DIR) -> Tuple[Optional[Path], bool]:
    """
    Locates dataset folder.
    Returns tuple: (dataset_path, is_sample_flag)
    
    Priority Order:
    1. RAW_DATA_DIR / GTSRB / Train (or RAW_DATA_DIR / GTSRB)
    2. RAW_DATA_DIR / Train
    3. RAW_DATA_DIR (if numeric folders 0..42 exist)
    4. SAMPLE_DATA_DIR (Pipeline demo / unit test ONLY)
    """
    # 1. Check GTSRB subfolder in data/raw/
    if (GTSRB_RAW_DIR / "Train").exists() and any((GTSRB_RAW_DIR / "Train").glob("*")):
        return GTSRB_RAW_DIR / "Train", False
    if GTSRB_RAW_DIR.exists() and any(d.name.isdigit() for d in GTSRB_RAW_DIR.iterdir() if d.is_dir()):
        return GTSRB_RAW_DIR, False

    # 2. Check Train subfolder in data/raw/
    if (base_path / "Train").exists() and any((base_path / "Train").glob("*")):
        return base_path / "Train", False
    
    # 3. Check direct numeric folders in data/raw/
    if base_path.exists():
        subdirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]
        if len(subdirs) > 0:
            return base_path, False
    
    # 4. Fallback to Sample Dataset (FOR DEMO/TESTING PIPELINE ONLY)
    if SAMPLE_DATA_DIR.exists():
        sample_subdirs = [d for d in SAMPLE_DATA_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
        if len(sample_subdirs) > 0:
            logger.warning("Real GTSRB dataset not found in data/raw/. Using SAMPLE dataset for pipeline testing ONLY.")
            return SAMPLE_DATA_DIR, True
            
    # 5. Auto-create sample dataset on-the-fly for cloud deployments
    logger.warning("Dataset directory missing. Auto-generating sample demo dataset for deployment...")
    try:
        create_sample_dataset()
        if SAMPLE_DATA_DIR.exists():
            return SAMPLE_DATA_DIR, True
    except Exception as e:
        logger.error(f"Failed to auto-create sample dataset: {e}")

    return None, False


def load_raw_dataset(dataset_dir: Optional[Path] = None, max_samples_per_class: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    """
    Loads images and labels from dataset directory.
    Returns:
        images: List of np.ndarray (uint8 BGR/RGB images)
        labels: np.ndarray of shape (N,) int
        stats: dictionary with summary statistics (including 'is_sample')
    """
    if dataset_dir:
        target_dir = dataset_dir
        is_sample = (SAMPLE_DATA_DIR in target_dir.parents or target_dir == SAMPLE_DATA_DIR)
    else:
        target_dir, is_sample = find_dataset_dir()
    
    if target_dir is None or not target_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_DIR} or {SAMPLE_DATA_DIR}.\n"
            "Please place real GTSRB dataset in data/raw/GTSRB/Train/<class_id>/ or run:\n"
            "python -m src.data_loader --create-sample"
        )

    images = []
    labels = []
    class_counts = {}

    # Read class subfolders (formatted as '0', '1', ..., '42' or '00000', '00001', ..., '00042')
    class_dirs = sorted(
        [d for d in target_dir.iterdir() if d.is_dir() and (d.name.isdigit() or d.name.lstrip('0').isdigit() or d.name == '00000')],
        key=lambda x: int(x.name)
    )
    
    if not class_dirs:
        raise ValueError(f"No numeric class folders (0..42) found in {target_dir}")

    for c_dir in class_dirs:
        class_id = int(c_dir.name)
        img_paths = list(c_dir.glob("*.png")) + list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.ppm"))
        
        if max_samples_per_class:
            img_paths = img_paths[:max_samples_per_class]

        count = 0
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
                labels.append(class_id)
                count += 1

        class_counts[class_id] = count

    labels = np.array(labels, dtype=int)

    stats = {
        "total_images": len(images),
        "total_classes": len(class_counts),
        "class_counts": class_counts,
        "dataset_path": str(target_dir),
        "is_sample": is_sample
    }

    dataset_type_str = "SAMPLE/DEMO DATASET" if is_sample else "REAL GTSRB DATASET"
    logger.info(f"Loaded {len(images)} images across {len(class_counts)} classes from {target_dir} ({dataset_type_str})")
    return images, labels, stats


def generate_sample_dataset(output_dir: Path = SAMPLE_DATA_DIR, samples_per_class: int = 20) -> None:
    """
    Generates clean synthetic traffic sign images for all 43 GTSRB classes
    strictly for pipeline validation and UI demonstration purposes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating synthetic sample dataset in {output_dir}...")
    
    np.random.seed(42)
    img_size = 64

    for class_id in range(NUM_CLASSES):
        c_dir = output_dir / str(class_id)
        c_dir.mkdir(parents=True, exist_ok=True)
        
        if class_id in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17]:
            base_bg = (240, 240, 240)
            border_color = (220, 20, 20)
            shape = "circle"
        elif class_id in [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
            base_bg = (240, 240, 240)
            border_color = (220, 20, 20)
            shape = "triangle"
        elif class_id in [33, 34, 35, 36, 37, 38, 39, 40]:
            base_bg = (30, 90, 220)
            border_color = (255, 255, 255)
            shape = "blue_circle"
        elif class_id == 14:
            base_bg = (220, 20, 20)
            border_color = (255, 255, 255)
            shape = "octagon"
        elif class_id == 13:
            base_bg = (240, 240, 240)
            border_color = (220, 20, 20)
            shape = "inv_triangle"
        else:
            base_bg = (200, 200, 200)
            border_color = (50, 50, 50)
            shape = "rect"

        for i in range(samples_per_class):
            img = np.full((img_size, img_size, 3), base_bg, dtype=np.uint8)
            
            if shape == "circle":
                cv2.circle(img, (32, 32), 28, border_color, 6)
                cv2.putText(img, str(class_id), (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            elif shape == "triangle":
                pts = np.array([[32, 6], [6, 56], [58, 56]], np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=5)
                cv2.putText(img, str(class_id), (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            elif shape == "blue_circle":
                cv2.circle(img, (32, 32), 28, (30, 90, 220), -1)
                cv2.circle(img, (32, 32), 28, border_color, 3)
                cv2.putText(img, str(class_id), (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif shape == "octagon":
                pts = np.array([[20, 6], [44, 6], [58, 20], [58, 44], [44, 58], [20, 58], [6, 44], [6, 20]], np.int32)
                cv2.fillPoly(img, [pts], color=(220, 20, 20))
                cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=3)
                cv2.putText(img, "STOP", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            elif shape == "inv_triangle":
                pts = np.array([[6, 10], [58, 10], [32, 58]], np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=5)
                cv2.putText(img, "YLD", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            else:
                cv2.rectangle(img, (8, 8), (56, 56), border_color, 4)
                cv2.putText(img, str(class_id), (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            noise = np.random.normal(0, 8, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            save_path = c_dir / f"sample_{i:03d}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    logger.info(f"Successfully generated {samples_per_class * NUM_CLASSES} sample images in {output_dir}")


def generate_external_test_samples(output_dir: Path = EXTERNAL_TEST_DIR) -> List[Path]:
    """
    Creates a set of sample external traffic sign test images in data/external_test/
    to test model generalization on unseen camera images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    test_configs = [
        {"class_id": 14, "filename": "ext_stop_sign.png", "text": "STOP", "bg": (220, 20, 20), "shape": "octagon"},
        {"class_id": 13, "filename": "ext_yield_sign.png", "text": "YIELD", "bg": (240, 240, 240), "shape": "inv_triangle"},
        {"class_id": 17, "filename": "ext_no_entry.png", "text": "NO ENTRY", "bg": (220, 20, 20), "shape": "circle_line"},
        {"class_id": 1, "filename": "ext_speed_30.png", "text": "30", "bg": (240, 240, 240), "shape": "speed_limit"},
        {"class_id": 38, "filename": "ext_keep_right.png", "text": "KEEP R", "bg": (30, 90, 220), "shape": "blue_arrow"}
    ]

    for cfg in test_configs:
        file_path = output_dir / cfg["filename"]
        if not file_path.exists():
            img = np.full((64, 64, 3), cfg["bg"], dtype=np.uint8)
            if cfg["shape"] == "octagon":
                pts = np.array([[20, 6], [44, 6], [58, 20], [58, 44], [44, 58], [20, 58], [6, 44], [6, 20]], np.int32)
                cv2.fillPoly(img, [pts], color=(220, 20, 20))
                cv2.polylines(img, [pts], True, (255, 255, 255), 3)
                cv2.putText(img, "STOP", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            elif cfg["shape"] == "speed_limit":
                cv2.circle(img, (32, 32), 28, (220, 20, 20), 6)
                cv2.putText(img, "30", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            elif cfg["shape"] == "circle_line":
                cv2.circle(img, (32, 32), 28, (220, 20, 20), -1)
                cv2.rectangle(img, (12, 28), (52, 36), (255, 255, 255), -1)
            elif cfg["shape"] == "inv_triangle":
                pts = np.array([[6, 10], [58, 10], [32, 58]], np.int32)
                cv2.polylines(img, [pts], True, (220, 20, 20), 5)
                cv2.putText(img, "YLD", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            else:
                cv2.circle(img, (32, 32), 28, (30, 90, 220), -1)
                cv2.arrowedLine(img, (18, 44), (44, 18), (255, 255, 255), 5, tipLength=0.4)

            cv2.imwrite(str(file_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        generated_paths.append(file_path)

    return generated_paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Loader & Sample Generator")
    parser.add_argument("--create-sample", action="store_true", help="Generate synthetic sample dataset")
    parser.add_argument("--create-external", action="store_true", help="Generate sample external test images")
    args = parser.parse_args()

    if args.create_sample:
        generate_sample_dataset()
    elif args.create_external:
        ext_imgs = generate_external_test_samples()
        print(f"Generated {len(ext_imgs)} external test images in {EXTERNAL_TEST_DIR}")
    else:
        try:
            imgs, lbls, st = load_raw_dataset()
            print(f"Loaded dataset stats: {st}")
        except FileNotFoundError as e:
            print(e)
