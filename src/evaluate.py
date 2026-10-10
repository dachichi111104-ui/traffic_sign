import cv2
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
import logging

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

from src.config import (
    CNN_MODEL_PATH, SVM_MODEL_PATH, RESULT_DIR, MISCLASSIFIED_DIR,
    GTSRB_CLASSES, RANDOM_STATE, TEST_RATIO, VAL_RATIO,
    CONFUSION_MATRIX_CNN_PATH, CONFUSION_MATRIX_SVM_PATH, COMPARISON_CSV_PATH
)
from src.data_loader import load_raw_dataset
from src.preprocessing import split_data, resize_image, normalize_pixels
from src.feature_extraction import extract_hog_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: Path):
    """
    Generates and saves Confusion Matrix heatmap.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True, ax=ax, square=True)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def analyze_misclassifications(
    X_test_raw: List[np.ndarray],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    max_save: int = 15
) -> List[Dict]:
    """
    Identifies misclassified images, calculates top confused pairs,
    and saves sample misclassified images to results/misclassified/
    """
    MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

    # Find misclassified indices
    mis_indices = np.where(y_test != y_pred)[0]
    logger.info(f"Found {len(mis_indices)} misclassified samples out of {len(y_test)} test samples.")

    misclassified_list = []
    
    for count, idx in enumerate(mis_indices):
        true_cls = int(y_test[idx])
        pred_cls = int(y_pred[idx])
        conf = float(np.max(y_pred_probs[idx]) * 100.0)

        true_info = GTSRB_CLASSES.get(true_cls, {"vi": f"Class {true_cls}", "en": f"Class {true_cls}"})
        pred_info = GTSRB_CLASSES.get(pred_cls, {"vi": f"Class {pred_cls}", "en": f"Class {pred_cls}"})

        sample_name = f"misclassified_{count:03d}_true_{true_cls}_pred_{pred_cls}.png"
        sample_path = MISCLASSIFIED_DIR / sample_name

        if count < max_save:
            img_rgb = X_test_raw[idx]
            cv2.imwrite(str(sample_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        misclassified_list.append({
            "index": int(idx),
            "true_class": true_cls,
            "true_name_vi": true_info["vi"],
            "true_name_en": true_info["en"],
            "pred_class": pred_cls,
            "pred_name_vi": pred_info["vi"],
            "pred_name_en": pred_info["en"],
            "confidence": round(conf, 2),
            "image_filename": sample_name if count < max_save else ""
        })

    # Save summary json
    with open(MISCLASSIFIED_DIR / "misclassified_info.json", "w", encoding="utf-8") as f:
        json.dump(misclassified_list, f, indent=4, ensure_ascii=False)

    return misclassified_list


def find_top_confused_pairs(y_test: np.ndarray, y_pred: np.ndarray, top_n: int = 5) -> List[Dict]:
    """
    Finds class pairs that are most frequently confused with each other.
    """
    cm = confusion_matrix(y_test, y_pred)
    np.fill_diagonal(cm, 0)  # Ignore correct predictions
    
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                true_info = GTSRB_CLASSES.get(i, {"vi": f"Class {i}", "en": f"Class {i}"})
                pred_info = GTSRB_CLASSES.get(j, {"vi": f"Class {j}", "en": f"Class {j}"})
                pairs.append({
                    "true_class": i,
                    "true_name": true_info["vi"],
                    "pred_class": j,
                    "pred_name": pred_info["vi"],
                    "count": int(cm[i, j])
                })
                
    pairs = sorted(pairs, key=lambda x: x["count"], reverse=True)[:top_n]
    return pairs


def evaluate_models(dataset_dir: Path = None) -> Dict:
    """
    Evaluates both CNN and HOG+SVM models strictly on held-out Test set.
    Outputs:
    - classification_report.json
    - metrics.json
    - model_comparison.csv
    - confusion_matrix_cnn.png
    - confusion_matrix_svm.png
    - misclassified images & json
    """
from src.config import (
    CNN_MODEL_PATH, SVM_MODEL_PATH, RESULT_DIR, MISCLASSIFIED_DIR,
    GTSRB_CLASSES, RANDOM_STATE, TEST_RATIO, VAL_RATIO, MODEL_METADATA_PATH,
    CONFUSION_MATRIX_CNN_PATH, CONFUSION_MATRIX_SVM_PATH, COMPARISON_CSV_PATH
)
from src.data_loader import load_raw_dataset
from src.preprocessing import split_data, resize_image, preprocess_dataset_batch
from src.feature_extraction import extract_hog_batch


def evaluate_models(dataset_dir: Path = None) -> Dict:
    """
    Evaluates both CNN and HOG+SVM models strictly on held-out Test set.
    Reads preprocessing_config from model_metadata.json to ensure exact test pipeline alignment.
    """
    logger.info("--- Starting Comprehensive Model Evaluation on Test Set ---")
    
    # 1. Load Dataset & Metadata Configuration
    images_raw, labels, stats = load_raw_dataset(dataset_dir=dataset_dir)

    cfg = {
        "use_clahe": True,
        "use_gamma": False,
        "use_unsharp": False,
        "norm_type": "minmax",
        "train_mean": None,
        "train_std": None
    }
    if MODEL_METADATA_PATH.exists():
        try:
            with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                cfg.update(meta.get("preprocessing_config", {}))
                logger.info(f"Loaded model preprocessing metadata config: {cfg}")
        except Exception as e:
            logger.warning(f"Could not read model metadata: {e}. Using default preprocessing config.")

    # 2. Strict Train/Val/Test Split (70/15/15) BEFORE any augmentation or sampling
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(
        images_raw, labels, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, random_state=RANDOM_STATE
    )

    # 3. Preprocess Test Set using exact trained model configuration & train_mean/train_std
    X_test_norm, _, _ = preprocess_dataset_batch(
        X_test_raw,
        use_clahe=cfg.get("use_clahe", True),
        use_gamma=cfg.get("use_gamma", False),
        use_unsharp=cfg.get("use_unsharp", False),
        norm_type=cfg.get("norm_type", "minmax"),
        mean=cfg.get("train_mean"),
        std=cfg.get("train_std")
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    results_summary = {}

    # --- 3. Evaluate CNN Model ---
    if CNN_MODEL_PATH.exists():
        logger.info(f"Loading CNN model from {CNN_MODEL_PATH}...")
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
        
        y_pred_probs_cnn = cnn_model.predict(X_test_norm, verbose=0)
        y_pred_cnn = np.argmax(y_pred_probs_cnn, axis=1)

        acc_cnn = float(accuracy_score(y_test, y_pred_cnn))
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred_cnn, average='weighted', zero_division=0)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred_cnn, average='macro', zero_division=0)

        cnn_metrics = {
            "model": "CNN",
            "accuracy": round(acc_cnn, 4),
            "precision_weighted": round(float(prec_w), 4),
            "recall_weighted": round(float(rec_w), 4),
            "f1_score_weighted": round(float(f1_w), 4),
            "precision_macro": round(float(prec_m), 4),
            "recall_macro": round(float(rec_m), 4),
            "f1_score_macro": round(float(f1_m), 4),
            "num_test_samples": len(y_test),
            "is_sample_dataset": stats.get("is_sample", False)
        }
        results_summary["CNN"] = cnn_metrics

        comparison_rows.append({
            "Model": "CNN",
            "Accuracy": f"{acc_cnn*100:.2f}%",
            "Precision (Weighted)": f"{prec_w*100:.2f}%",
            "Recall (Weighted)": f"{rec_w*100:.2f}%",
            "F1-score (Weighted)": f"{f1_w*100:.2f}%",
            "Precision (Macro)": f"{prec_m*100:.2f}%",
            "Recall (Macro)": f"{rec_m*100:.2f}%",
            "F1-score (Macro)": f"{f1_m*100:.2f}%"
        })

        # Save classification report
        cls_report = classification_report(y_test, y_pred_cnn, output_dict=True, zero_division=0)
        with open(RESULT_DIR / "classification_report.json", "w", encoding="utf-8") as f:
            json.dump(cls_report, f, indent=4)

        # Plot CNN confusion matrix
        plot_confusion_matrix(
            y_test, y_pred_cnn,
            title="CNN Traffic Sign Classification - Confusion Matrix",
            save_path=CONFUSION_MATRIX_CNN_PATH
        )

        # Misclassified & Confused pairs analysis
        top_confused = find_top_confused_pairs(y_test, y_pred_cnn, top_n=5)
        cnn_metrics["top_confused_pairs"] = top_confused
        mis_list = analyze_misclassifications(X_test_raw, y_test, y_pred_cnn, y_pred_probs_cnn)
        cnn_metrics["num_misclassified"] = len(mis_list)

        logger.info(f"CNN Evaluation -> Accuracy: {acc_cnn:.4f}, Weighted F1: {f1_w:.4f}")
    else:
        logger.warning(f"CNN model file not found at {CNN_MODEL_PATH}")

    # --- 4. Evaluate Baseline HOG + SVM Model ---
    if SVM_MODEL_PATH.exists():
        logger.info(f"Loading SVM model from {SVM_MODEL_PATH}...")
        svm_model = joblib.load(SVM_MODEL_PATH)
        
        X_test_hog = extract_hog_batch(X_test_raw)
        y_pred_svm = svm_model.predict(X_test_hog)
        y_pred_probs_svm = svm_model.predict_proba(X_test_hog)

        acc_svm = float(accuracy_score(y_test, y_pred_svm))
        prec_w_s, rec_w_s, f1_w_s, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='weighted', zero_division=0)
        prec_m_s, rec_m_s, f1_m_s, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='macro', zero_division=0)

        svm_metrics = {
            "model": "HOG + SVM",
            "accuracy": round(acc_svm, 4),
            "precision_weighted": round(float(prec_w_s), 4),
            "recall_weighted": round(float(rec_w_s), 4),
            "f1_score_weighted": round(float(f1_w_s), 4),
            "precision_macro": round(float(prec_m_s), 4),
            "recall_macro": round(float(rec_m_s), 4),
            "f1_score_macro": round(float(f1_m_s), 4),
            "num_test_samples": len(y_test),
            "is_sample_dataset": stats.get("is_sample", False)
        }
        results_summary["HOG + SVM"] = svm_metrics

        comparison_rows.append({
            "Model": "HOG + SVM",
            "Accuracy": f"{acc_svm*100:.2f}%",
            "Precision (Weighted)": f"{prec_w_s*100:.2f}%",
            "Recall (Weighted)": f"{rec_w_s*100:.2f}%",
            "F1-score (Weighted)": f"{f1_w_s*100:.2f}%",
            "Precision (Macro)": f"{prec_m_s*100:.2f}%",
            "Recall (Macro)": f"{rec_m_s*100:.2f}%",
            "F1-score (Macro)": f"{f1_m_s*100:.2f}%"
        })

        # Plot SVM confusion matrix
        plot_confusion_matrix(
            y_test, y_pred_svm,
            title="HOG + SVM Baseline - Confusion Matrix",
            save_path=CONFUSION_MATRIX_SVM_PATH
        )

        logger.info(f"SVM Evaluation -> Accuracy: {acc_svm:.4f}, Weighted F1: {f1_w_s:.4f}")

    # 5. Save Model Comparison CSV
    if comparison_rows:
        df_comp = pd.DataFrame(comparison_rows)
        df_comp.to_csv(COMPARISON_CSV_PATH, index=False, encoding="utf-8")
        logger.info(f"Saved model comparison table to {COMPARISON_CSV_PATH}")

    # 6. Save summary metrics
    with open(RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_summary.get("CNN", {}), f, indent=4, ensure_ascii=False)

    with open(RESULT_DIR / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)

    logger.info("Evaluation complete! All evaluation outputs saved to results/ folder.")
    return results_summary


if __name__ == "__main__":
    evaluate_models()
