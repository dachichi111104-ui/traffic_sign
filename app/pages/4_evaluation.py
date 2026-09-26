import sys
import json
import streamlit as st
from PIL import Image
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    RESULT_DIR, MISCLASSIFIED_DIR, CONFUSION_MATRIX_CNN_PATH, CONFUSION_MATRIX_SVM_PATH,
    TRAINING_ACCURACY_PATH, TRAINING_LOSS_PATH, METRICS_JSON_PATH
)
from app.components.ui_components import apply_custom_theme, render_hero_banner, render_metric_card, render_alert

st.set_page_config(page_title="Model Evaluation | Traffic Sign Classification", layout="wide")
apply_custom_theme()

render_hero_banner(
    title="Đánh Giá Chi Tiết Mô Hình (Model Evaluation)",
    subtitle="Trực quan hóa kết quả thực nghiệm trên tập kiểm thử Test Set độc lập (Accuracy, F1-score, Confusion Matrix, Training Curves & Misclassified Images).",
    badge_text="BÁO CÁO THỰC NGHIỆM"
)

if not METRICS_JSON_PATH.exists():
    render_alert("CHƯA CÓ KẾT QUẢ THỰC NGHIỆM: Vui lòng huấn luyện mô hình và chạy: python -m src.evaluate", "warning")
    st.stop()

with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

is_sample = metrics.get("is_sample_dataset", False)
if is_sample:
    render_alert("CHÚ Ý: Kết quả đánh giá dưới đây được sinh từ Sample Demo Dataset. Kết quả thực nghiệm GTSRB thật sẽ tự động cập nhật sau khi bạn huấn luyện trên dataset thật.", "warning")

# Top Metric Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Accuracy (Độ chính xác)", f"{metrics.get('accuracy', 0)*100:.2f}%", "Overall Test Accuracy")
with c2:
    render_metric_card("F1-Score (Weighted)", f"{metrics.get('f1_score_weighted', metrics.get('f1_score', 0))*100:.2f}%", "Weighted F1 Average")
with c3:
    render_metric_card("F1-Score (Macro)", f"{metrics.get('f1_score_macro', 0)*100:.2f}%", "Macro F1 Average")
with c4:
    render_metric_card("Số Mẫu Test Set", str(metrics.get("num_test_samples", 0)), "Held-out Test Samples")

tab_metrics, tab_curves, tab_misclassified = st.tabs([
    "1. Metrics & Confusion Matrix",
    "2. Training Curves (Accuracy & Loss)",
    "3. Misclassified Images & Confused Pairs"
])

# --- TAB 1: METRICS & CONFUSION MATRIX ---
with tab_metrics:
    st.markdown("<div class='section-header'>Ma Trận Nhầm Lẫn (Confusion Matrix) Trên Tập Test</div>", unsafe_allow_html=True)
    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.markdown("### Mô Hình CNN Deep Learning")
        if CONFUSION_MATRIX_CNN_PATH.exists():
            img_cm_cnn = Image.open(CONFUSION_MATRIX_CNN_PATH)
            st.image(img_cm_cnn, caption="CNN Confusion Matrix", use_container_width=True)
        else:
            render_alert("Chưa có ảnh confusion_matrix_cnn.png", "info")

    with col_cm2:
        st.markdown("### Mô Hình HOG + SVM Baseline")
        if CONFUSION_MATRIX_SVM_PATH.exists():
            img_cm_svm = Image.open(CONFUSION_MATRIX_SVM_PATH)
            st.image(img_cm_svm, caption="SVM Baseline Confusion Matrix", use_container_width=True)
        else:
            render_alert("Chưa có ảnh confusion_matrix_svm.png", "info")

    report_file = RESULT_DIR / "classification_report.json"
    if report_file.exists():
        st.markdown("<div class='section-header'>Báo Cáo Phân Loại Chi Tiết (Classification Report)</div>", unsafe_allow_html=True)
        with open(report_file, "r", encoding="utf-8") as rf:
            cls_report = json.load(rf)
            st.json(cls_report)

# --- TAB 2: TRAINING CURVES ---
with tab_curves:
    st.markdown("<div class='section-header'>Đường Cong Huấn Luyện (Training vs Validation)</div>", unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("### Biểu Đồ Accuracy")
        if TRAINING_ACCURACY_PATH.exists():
            img_acc = Image.open(TRAINING_ACCURACY_PATH)
            st.image(img_acc, caption="Training vs Validation Accuracy per Epoch", use_container_width=True)
        else:
            render_alert("Chưa có biểu đồ training_accuracy.png", "info")

    with col_c2:
        st.markdown("### Biểu Đồ Loss")
        if TRAINING_LOSS_PATH.exists():
            img_loss = Image.open(TRAINING_LOSS_PATH)
            st.image(img_loss, caption="Training vs Validation Loss per Epoch", use_container_width=True)
        else:
            render_alert("Chưa có biểu đồ training_loss.png", "info")

# --- TAB 3: MISCLASSIFIED IMAGES ---
with tab_misclassified:
    st.markdown("<div class='section-header'>Phân Tích Các Trường Hợp Dự Đoán Sai (Misclassified Images)</div>", unsafe_allow_html=True)

    top_confused = metrics.get("top_confused_pairs", [])
    if top_confused:
        st.markdown("#### Top Cặp Lớp Thường Bị Nhầm Lẫn Nhất:")
        for pair in top_confused:
            st.write(f"• **True:** `{pair['true_name']}` (ID {pair['true_class']}) -> **Predicted:** `{pair['pred_name']}` (ID {pair['pred_class']}): **{pair['count']} mẫu**")

    st.markdown("---")
    st.markdown("#### Xem Trực Quan Các Mẫu Bị Phân Loại Sai:")

    mis_info_file = MISCLASSIFIED_DIR / "misclassified_info.json"
    if mis_info_file.exists():
        with open(mis_info_file, "r", encoding="utf-8") as mf:
            mis_data = json.load(mf)
            
        if not mis_data:
            render_alert("Không có mẫu nào bị phân loại sai trên tập Test Set.", "success")
        else:
            cols = st.columns(min(len(mis_data), 4))
            for i, mis in enumerate(mis_data[:8]):
                col_idx = i % 4
                if i > 0 and col_idx == 0:
                    cols = st.columns(min(len(mis_data) - i, 4))
                
                with cols[col_idx]:
                    img_path = MISCLASSIFIED_DIR / mis["image_filename"]
                    if img_path.exists():
                        st.image(Image.open(img_path), caption=f"Index {mis['index']}", use_container_width=True)
                        st.write(f"**True:** {mis['true_name_vi']} (ID {mis['true_class']})")
                        st.write(f"**Pred:** {mis['pred_name_vi']} (ID {mis['pred_class']})")
                        st.write(f"**Conf:** {mis['confidence']:.2f}%")
    else:
        render_alert("Chưa có danh sách misclassified_info.json.", "info")
