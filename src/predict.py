import os
import sys
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Union, List, Tuple
import logging

import tensorflow as tf

from src.config import (
    CNN_MODEL_PATH, CNN_CUSTOM_AUG_PATH, CNN_MOBILENET_NO_AUG_PATH, CNN_MOBILENET_AUG_PATH,
    SVM_MODEL_PATH, SVM_NO_AUG_PATH, SVM_AUG_PATH,
    CLASS_MAPPING_PATH, GTSRB_CLASSES, IMAGE_SIZE, EXTERNAL_TEST_DIR, MODEL_METADATA_PATH
)
from src.preprocessing import prepare_image_pipeline, load_image, resize_image
from src.feature_extraction import extract_hog_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model cache
_CUSTOM_CNN_MODEL = None
_MOBILENET_MODEL = None
_SVM_MODEL = None


def get_cnn_model():
    """Gets primary Custom CNN model (with augmentation)."""
    global _CUSTOM_CNN_MODEL
    if _CUSTOM_CNN_MODEL is None:
        target_path = CNN_CUSTOM_AUG_PATH if CNN_CUSTOM_AUG_PATH.exists() else CNN_MODEL_PATH
        if not target_path.exists():
            target_path = CNN_MODEL_PATH
        logger.info(f"Loading Custom CNN model from {target_path}")
        _CUSTOM_CNN_MODEL = tf.keras.models.load_model(target_path)
    return _CUSTOM_CNN_MODEL


def get_mobilenet_model():
    """Gets MobileNetV2 model for transfer learning comparison."""
    global _MOBILENET_MODEL
    if _MOBILENET_MODEL is None:
        target_path = CNN_MOBILENET_NO_AUG_PATH if CNN_MOBILENET_NO_AUG_PATH.exists() else CNN_MOBILENET_AUG_PATH
        if not target_path.exists():
            from src.train_cnn import build_transfer_learning_model
            _MOBILENET_MODEL = build_transfer_learning_model()
            _MOBILENET_MODEL.save(CNN_MOBILENET_NO_AUG_PATH)
        else:
            logger.info(f"Loading MobileNetV2 model from {target_path}")
            _MOBILENET_MODEL = tf.keras.models.load_model(target_path)
    return _MOBILENET_MODEL


def get_svm_model():
    """Gets HOG + SVM model."""
    global _SVM_MODEL
    if _SVM_MODEL is None:
        target_path = SVM_NO_AUG_PATH if SVM_NO_AUG_PATH.exists() else SVM_MODEL_PATH
        if not target_path.exists():
            logger.warning(f"SVM model not found at {target_path}. Auto-training baseline model for deployment...")
            from src.train_svm import train_svm_baseline
            train_svm_baseline()
        logger.info(f"Loading SVM model from {target_path}")
        _SVM_MODEL = joblib.load(target_path)
    return _SVM_MODEL


def get_model_preprocessing_config() -> Dict:
    if MODEL_METADATA_PATH.exists():
        try:
            with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                return meta.get("preprocessing_config", {})
        except Exception:
            pass
    return {
        "use_clahe": True,
        "use_gamma": False,
        "use_unsharp": False,
        "norm_type": "minmax",
        "train_mean": None,
        "train_std": None
    }


def predict_traffic_sign(
    image_input: Union[str, Path, np.ndarray, bytes],
    model_type: str = "Custom CNN"
) -> Dict:
    """
    Inference pipeline:
    Image -> Preprocessing (matching model metadata config) -> Model -> Softmax/Probabilities -> Structured Output
    Supports: "Custom CNN", "MobileNetV2", "HOG + SVM"
    """
    img_rgb = load_image(image_input)
    
    m_type_lower = model_type.strip().lower()
    if "mobilenet" in m_type_lower:
        model = get_mobilenet_model()
        cfg = get_model_preprocessing_config()
        processed_tensor = prepare_image_pipeline(img_rgb, target_size=IMAGE_SIZE, preprocessing_config=cfg)
        probs = model.predict(processed_tensor, verbose=0)[0]
    elif "svm" in m_type_lower or "hog" in m_type_lower:
        model = get_svm_model()
        resized = resize_image(img_rgb, IMAGE_SIZE)
        hog_feat = extract_hog_features(resized).reshape(1, -1)
        probs = model.predict_proba(hog_feat)[0]
    else:
        # Default: Custom CNN
        model = get_cnn_model()
        cfg = get_model_preprocessing_config()
        processed_tensor = prepare_image_pipeline(img_rgb, target_size=IMAGE_SIZE, preprocessing_config=cfg)
        probs = model.predict(processed_tensor, verbose=0)[0]

    # Rank predictions by probability descending
    top_indices = np.argsort(probs)[::-1]
    
    top_3 = []
    for idx in top_indices[:3]:
        class_id = int(idx)
        class_info = GTSRB_CLASSES.get(class_id, {"vi": f"Biển số {class_id}", "en": f"Class {class_id}"})
        conf = float(probs[idx] * 100.0)
        top_3.append({
            "class_id": class_id,
            "name_vi": class_info["vi"],
            "name_en": class_info["en"],
            "confidence": round(conf, 2)
        })

    best_pred = top_3[0]

    return {
        "class_id": best_pred["class_id"],
        "name_vi": best_pred["name_vi"],
        "name_en": best_pred["name_en"],
        "confidence": best_pred["confidence"],
        "top_3": top_3,
        "all_probabilities": probs.tolist()
    }


def predict_external_test_folder(folder_path: Path = EXTERNAL_TEST_DIR, model_type: str = "CNN") -> List[Dict]:
    """
    Runs prediction on all external camera test images inside folder_path.
    """
    if not folder_path.exists():
        return []

    image_files = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg"))
    results = []

    for img_file in image_files:
        try:
            pred = predict_traffic_sign(img_file, model_type=model_type)
            pred["file_name"] = img_file.name
            pred["file_path"] = str(img_file)
            results.append(pred)
        except Exception as e:
            logger.warning(f"Failed to predict external image {img_file.name}: {e}")

    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
        res = predict_traffic_sign(test_img_path)
        print(json.dumps(res, indent=4, ensure_ascii=False))
    else:
        ext_res = predict_external_test_folder()
        print(f"Predicted {len(ext_res)} external images.")
