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
    CNN_MODEL_PATH, SVM_MODEL_PATH, CLASS_MAPPING_PATH, GTSRB_CLASSES, IMAGE_SIZE, EXTERNAL_TEST_DIR
)
from src.preprocessing import prepare_image_pipeline, load_image, resize_image
from src.feature_extraction import extract_hog_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model cache
_CNN_MODEL = None
_SVM_MODEL = None


def get_cnn_model():
    global _CNN_MODEL
    if _CNN_MODEL is None:
        if not CNN_MODEL_PATH.exists():
            logger.warning(f"CNN model not found at {CNN_MODEL_PATH}. Auto-training model for deployment...")
            from src.train_cnn import train_cnn
            train_cnn(retrain=True)
        logger.info(f"Loading CNN model from {CNN_MODEL_PATH}")
        _CNN_MODEL = tf.keras.models.load_model(CNN_MODEL_PATH)
    return _CNN_MODEL


def get_svm_model():
    global _SVM_MODEL
    if _SVM_MODEL is None:
        if not SVM_MODEL_PATH.exists():
            logger.warning(f"SVM model not found at {SVM_MODEL_PATH}. Auto-training baseline model for deployment...")
            from src.train_svm import train_svm_baseline
            train_svm_baseline()
        logger.info(f"Loading SVM model from {SVM_MODEL_PATH}")
        _SVM_MODEL = joblib.load(SVM_MODEL_PATH)
    return _SVM_MODEL


def predict_traffic_sign(
    image_input: Union[str, Path, np.ndarray, bytes],
    model_type: str = "CNN"
) -> Dict:
    """
    Inference pipeline:
    Image -> Preprocessing -> Model -> Softmax/Probabilities -> Structured Output
    
    Returns dictionary with:
        - class_id: int
        - name_vi: str
        - name_en: str
        - confidence: float (0 - 100 %)
        - top_3: List[Dict] with class_id, name_vi, name_en, confidence
    """
    img_rgb = load_image(image_input)
    
    if model_type.upper() == "CNN":
        model = get_cnn_model()
        processed_tensor = prepare_image_pipeline(img_rgb, target_size=IMAGE_SIZE)
        probs = model.predict(processed_tensor, verbose=0)[0]
    elif model_type.upper() in ["SVM", "HOG + SVM"]:
        model = get_svm_model()
        resized = resize_image(img_rgb, IMAGE_SIZE)
        hog_feat = extract_hog_features(resized).reshape(1, -1)
        probs = model.predict_proba(hog_feat)[0]
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

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
