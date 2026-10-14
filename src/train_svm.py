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
import argparse

from src.config import (
    SVM_MODEL_PATH, SVM_NO_AUG_PATH, SVM_AUG_PATH, RESULT_DIR,
    CONFUSION_MATRIX_SVM_PATH, RANDOM_STATE, NUM_CLASSES
)
from src.data_loader import load_raw_dataset
from src.preprocessing import split_data, resize_image, augment_single_image
from src.feature_extraction import extract_hog_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_svm_baseline(dataset_dir: Path = None, save_model: bool = True, augment: bool = False):
    """
    Trains HOG + SVM model with optional data augmentation (applied strictly to Train set).
    """
    mode_str = "WITH Data Augmentation (Stage 2)" if augment else "WITHOUT Augmentation (Stage 1 Baseline)"
    logger.info(f"--- Starting HOG + SVM Model Training [{mode_str}] ---")
    
    # 1. Load Data
    images_raw, labels, stats = load_raw_dataset(dataset_dir=dataset_dir)
    
    # Resize all images to 32x32
    images_resized = [resize_image(img, (32, 32)) for img in images_raw]
    
    # 2. Split Data (Train 70%, Val 15%, Test 15%) BEFORE augmentation
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        images_resized, labels, test_ratio=0.15, val_ratio=0.15, random_state=RANDOM_STATE
    )

    # 3. Optional Augmentation strictly on Train set (Stage 2 requirement)
    if augment:
        logger.info("Applying data augmentation STRICTLY to SVM training set...")
        X_train_aug = []
        y_train_aug = []
        for img, lbl in zip(X_train, y_train):
            X_train_aug.append(img)
            y_train_aug.append(lbl)
            X_train_aug.append(augment_single_image(img))
            y_train_aug.append(lbl)
        X_train_final = np.array(X_train_aug, dtype=np.uint8)
        y_train_final = np.array(y_train_aug, dtype=int)
    else:
        X_train_final = X_train
        y_train_final = y_train

    # 4. Extract HOG Features
    logger.info(f"Extracting HOG features for Train ({len(X_train_final)}), Val ({len(X_val)}), and Test ({len(X_test)}) sets...")
    X_train_hog = extract_hog_batch(X_train_final)
    X_val_hog = extract_hog_batch(X_val)
    X_test_hog = extract_hog_batch(X_test)
    
    logger.info(f"HOG feature dimension: {X_train_hog.shape[1]}")

    # 5. Train SVM Classifier
    logger.info("Training SVM Classifier (RBF Kernel)...")
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=RANDOM_STATE)
    svm_model.fit(X_train_hog, y_train_final)

    # 6. Evaluate on Test Set
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

    model_label = "HOG + SVM (Aug)" if augment else "HOG + SVM (No Aug)"
    metrics = {
        "model": model_label,
        "augmented": augment,
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

    logger.info(f"SVM [{model_label}] Results -> Accuracy: {acc:.4f}, Weighted F1: {f1_weighted:.4f}, Macro F1: {f1_macro:.4f}")

    # 7. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'{model_label} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = RESULT_DIR / f"confusion_matrix_svm{'_aug' if augment else ''}.png"
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Save Model & Metrics
    if save_model:
        target_model_path = SVM_AUG_PATH if augment else SVM_NO_AUG_PATH
        target_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(svm_model, target_model_path)
        logger.info(f"SVM Model saved to {target_model_path}")

        # Also save to default SVM_MODEL_PATH if baseline or if desired
        if not augment:
            joblib.dump(svm_model, SVM_MODEL_PATH)

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        metrics_filename = "svm_aug_metrics.json" if augment else "svm_metrics.json"
        with open(RESULT_DIR / metrics_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
            
    return svm_model, metrics, (X_test_hog, y_test, y_pred, y_pred_proba)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HOG + SVM with or without Data Augmentation")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation to Train set before HOG extraction")
    args = parser.parse_args()
    
    train_svm_baseline(augment=args.augment)
