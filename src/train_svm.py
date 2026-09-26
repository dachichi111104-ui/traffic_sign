import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import logging

from src.config import (
    SVM_MODEL_PATH, RESULT_DIR, CONFUSION_MATRIX_SVM_PATH, RANDOM_STATE, NUM_CLASSES
)
from src.data_loader import load_raw_dataset
from src.preprocessing import split_data, resize_image
from src.feature_extraction import extract_hog_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_svm_baseline(dataset_dir: Path = None, save_model: bool = True):
    """
    Trains HOG + SVM baseline model.
    """
    logger.info("--- Starting HOG + SVM Baseline Model Training ---")
    
    # 1. Load Data
    images_raw, labels, stats = load_raw_dataset(dataset_dir=dataset_dir)
    
    # Resize all images to 32x32
    images_resized = [resize_image(img, (32, 32)) for img in images_raw]
    
    # 2. Split Data (Train 70%, Val 15%, Test 15%)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        images_resized, labels, test_ratio=0.15, val_ratio=0.15, random_state=RANDOM_STATE
    )

    # 3. Extract HOG Features
    logger.info("Extracting HOG features for Train, Val, and Test sets...")
    X_train_hog = extract_hog_batch(X_train)
    X_val_hog = extract_hog_batch(X_val)
    X_test_hog = extract_hog_batch(X_test)
    
    logger.info(f"HOG feature dimension: {X_train_hog.shape[1]}")

    # 4. Train SVM Classifier
    logger.info("Training SVM Classifier (RBF Kernel)...")
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=RANDOM_STATE)
    svm_model.fit(X_train_hog, y_train)

    # 5. Evaluate Baseline Model on Test Set
    logger.info("Evaluating HOG + SVM on Test Set...")
    y_pred = svm_model.predict(X_test_hog)
    y_pred_proba = svm_model.predict_proba(X_test_hog)
    
    acc = float(accuracy_score(y_test, y_pred))
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )

    metrics = {
        "model": "HOG + SVM",
        "accuracy": round(acc, 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_score_weighted": round(float(f1_weighted), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_score_macro": round(float(f1_macro), 4),
        "num_test_samples": len(y_test),
        "is_sample_dataset": stats.get("is_sample", False)
    }

    logger.info(f"SVM Results -> Accuracy: {acc:.4f}, Weighted F1: {f1_weighted:.4f}, Macro F1: {f1_macro:.4f}")

    # 6. Plot Confusion Matrix for SVM Baseline
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.title('HOG + SVM Baseline - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    CONFUSION_MATRIX_SVM_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_SVM_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Save Model & Metrics
    if save_model:
        SVM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(svm_model, SVM_MODEL_PATH)
        logger.info(f"SVM Model saved to {SVM_MODEL_PATH}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULT_DIR / "svm_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
            
    return svm_model, metrics, (X_test_hog, y_test, y_pred, y_pred_proba)


if __name__ == "__main__":
    train_svm_baseline()
