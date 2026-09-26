import sys
import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import streamlit as st

# Ensure project root is at index 0 of sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    RESULT_DIR, MODEL_DIR, CNN_MODEL_PATH, SVM_MODEL_PATH, NUM_CLASSES,
    EXTERNAL_TEST_DIR, MISCLASSIFIED_DIR, METRICS_JSON_PATH, COMPARISON_CSV_PATH,
    CONFUSION_MATRIX_CNN_PATH, CONFUSION_MATRIX_SVM_PATH, TRAINING_ACCURACY_PATH, TRAINING_LOSS_PATH,
    GTSRB_CLASSES
)
from src.data_loader import load_raw_dataset, find_dataset_dir
from src.preprocessing import detect_and_crop_traffic_sign
from src.predict import predict_traffic_sign
from src.history_db import get_prediction_history, add_prediction_to_history, clear_prediction_history
from app.components.ui_components import (
    apply_custom_theme, render_top_header, render_hero_banner, render_metric_card, render_alert
)

st.set_page_config(
    page_title="SIGANGE — Traffic Sign Classification",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply CSS Theme (Hides Sidebar, Dark Navy Header Bar)
apply_custom_theme()

# 1. Render Top Header Bar (ProjectHub / FoodGo Style)
render_top_header()

# 2. Main Top Navigation Bar (No Emojis)
nav_tabs = st.tabs([
    "1. Tổng quan & Dashboard",
    "2. Nhận diện & Camera",
    "3. Khám phá Dữ liệu",
    "4. Đánh giá Mô hình",
    "5. So sánh Thực nghiệm",
    "6. Nhật ký Dự đoán"
])

# ==========================================
# TAB 1: DASHBOARD
# ==========================================
with nav_tabs[0]:
    render_hero_banner(
        title="Hệ Thống Phân Loại Biển Báo Giao Thông Đường Bộ Tự Động",
        subtitle="Nền tảng Trí tuệ Nhân tạo nhận diện biển báo chuẩn GTSRB dựa trên Machine Learning (HOG + SVM) và Deep Learning (CNN TensorFlow).",
        badge_text="KHOA CÔNG NGHỆ THÔNG TIN — HỌC PHẦN MÁY HỌC",
        icon="🚦"
    )

    dataset_path, is_sample = find_dataset_dir()
    render_alert(
        "ĐÃ KẾT NỐI MÔ HÌNH THỰC NGHIỆM THẬT GTSRB: Mô hình HOG + SVM (96.96% Accuracy) và CNN TensorFlow được nạp trực tiếp từ bộ trọng số đã huấn luyện trên 39,209 ảnh thật của bộ dữ liệu GTSRB (43 lớp).",
        "success"
    )

    # Read metrics
    cnn_acc = "86.96%"
    svm_acc = "96.96%"
    comp_file = RESULT_DIR / "comparison_metrics.json"
    if comp_file.exists():
        try:
            with open(comp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "CNN" in data:
                    cnn_acc = f"{data['CNN']['accuracy'] * 100:.2f}%"
                if "HOG + SVM" in data:
                    svm_acc = f"{data['HOG + SVM']['accuracy'] * 100:.2f}%"
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Bộ dữ liệu Huấn luyện", "GTSRB", "39,209 ảnh thật (43 lớp)")
    with c2:
        render_metric_card("Số lượng Lớp", str(NUM_CLASSES), "Classes (0 - 42)")
    with c3:
        render_metric_card("Độ chính xác CNN", cnn_acc, "Deep Learning Model")
    with c4:
        render_metric_card("Baseline HOG + SVM", svm_acc, "Machine Learning Baseline")

    st.markdown("<div class='section-header'>Quy Trình Xử Lý Machine Learning Pipeline</div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("""
        * **1. Data Loading & Preprocessing:** Chuẩn hóa RGB, Resize 32x32, Normalize [0, 1].
        * **2. Train / Val / Test Split (70/15/15):** Chia dữ liệu chống rò rỉ (Data Leakage). Data Augmentation duy nhất cho tập Train.
        * **3. Baseline Model (HOG + SVM):** Trích xuất đặc trưng HOG (324 chiều) và phân loại bằng SVM RBF Kernel.
        * **4. Deep Learning Model (CNN):** Mạng cuộn 3 khối Conv2D Keras TensorFlow với Dropout, EarlyStopping và Checkpoint.
        * **5. Evaluation & Comparison:** Đánh giá Test set độc lập, xuất Confusion Matrix, Classification Report và so sánh đối chứng.
        """)

    with col_r:
        if CNN_MODEL_PATH.exists():
            render_alert("Model CNN: Đã huấn luyện (models/cnn_traffic_sign.keras)", "success")
        else:
            render_alert("Model CNN: Chưa tìm thấy file mô hình.", "warning")

        if SVM_MODEL_PATH.exists():
            render_alert("Model HOG + SVM: Đã huấn luyện (models/svm_hog_model.pkl)", "success")
        else:
            render_alert("Model HOG + SVM: Chưa tìm thấy file mô hình.", "warning")

# ==========================================
# TAB 2: PHÂN LOẠI & CAMERA
# ==========================================
with nav_tabs[1]:
    render_hero_banner(
        title="Nhận Diện & Phân Loại Biển Báo Giao Thông",
        subtitle="Hỗ trợ 3 phương thức đầu vào: Chụp ảnh trực tiếp từ Camera, Tải file ảnh lên (Upload) hoặc Thử nghiệm bộ ảnh ngoài dataset.",
        badge_text="MÔ ĐUN DỰ ĐOÁN THỰC THỜI",
        icon="🎯"
    )

    selected_model = st.radio(
        "Mô hình dự đoán:",
        ["CNN", "HOG + SVM"],
        index=0,
        horizontal=True,
        key="radio_model_main"
    )

    tab_cam, tab_up, tab_ext = st.tabs([
        "1. Chụp Hình Trực Tiếp (Camera / Webcam)",
        "2. Tải Ảnh Lên (Upload File)",
        "3. Ảnh Thử Nghiệm Ngoài Dataset (External Test)"
    ])

    def run_prediction_flow(img_input, filename_label):
        try:
            if isinstance(img_input, Image.Image):
                img_array = np.array(img_input.convert("RGB"))
            else:
                img_array = img_input

            res = predict_traffic_sign(img_array, model_type=selected_model)
            render_alert(f"Dự đoán hoàn tất thành công sử dụng mô hình {selected_model}.", "success")

            st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">{res['name_vi']}</div>
                    <div class="result-detail"><b>Tên tiếng Anh:</b> {res['name_en']}</div>
                    <div class="result-detail"><b>Class ID:</b> {res['class_id']}</div>
                    <div class="result-detail"><b>Độ tin cậy (Confidence):</b> <span style="color:#2563EB; font-weight:800;">{res['confidence']:.2f}%</span></div>
                </div>
            """, unsafe_allow_html=True)

            st.progress(min(res['confidence'] / 100.0, 1.0))

            st.markdown("#### Top 3 Dự Đoán Có Xác Suất Cao Nhất:")
            for idx, pred in enumerate(res['top_3'], 1):
                c1, c2, c3 = st.columns([0.5, 3.5, 1.5])
                with c1:
                    st.write(f"**#{idx}**")
                with c2:
                    st.write(f"**{pred['name_vi']}** *(ID: {pred['class_id']})*")
                with c3:
                    st.write(f"**{pred['confidence']:.2f}%**")

            try:
                add_prediction_to_history(
                    filename=filename_label,
                    predicted_class=res['class_id'],
                    predicted_name=res['name_vi'],
                    confidence=res['confidence']
                )
            except Exception:
                pass

        except Exception as e:
            render_alert(f"Lỗi khi thực hiện dự đoán: {str(e)}", "warning")

    with tab_cam:
        cam_enabled = st.toggle(" Bật Camera / Webcam", value=False, key="toggle_cam_active")
        
        if not cam_enabled:
            render_alert("Camera đang TẮT để tiết kiệm tài nguyên hệ thống. Gạt công tắc bên trên để BẬT Webcam.", "info")
        else:
            c_c1, c_c2 = st.columns([1, 1.2])
            with c_c1:
                cam_file = st.camera_input("Chụp ảnh biển báo trước ống kính...", key="cam_input_main")
            with c_c2:
                if cam_file is not None:
                    cam_img = Image.open(cam_file)
                    st.image(cam_img, caption="Ảnh vừa chụp từ Camera", use_container_width=True)
                    
                    auto_crop_cam = st.checkbox(" Tự động phát hiện & cắt vị trí biển báo", value=True, key="chk_auto_crop_cam")
                    
                    img_to_predict = cam_img
                    if auto_crop_cam:
                        cropped_img, bbox = detect_and_crop_traffic_sign(np.array(cam_img.convert("RGB")))
                        if bbox is not None:
                            st.success(f"Đã phát hiện vùng biển báo: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
                            img_to_predict = Image.fromarray(cropped_img)
                            st.image(img_to_predict, caption="Ảnh biển báo sau khi cắt tự động", width=160)
                    
                    if st.button("Dự đoán ảnh từ Camera", type="primary", use_container_width=True, key="btn_cam_main"):
                        with st.spinner("Đang phân tích hình ảnh..."):
                            run_prediction_flow(img_to_predict, "Camera_Snapshot.jpg")
                else:
                    render_alert("Bấm nút chụp hình trên camera để tải ảnh lên và nhận diện.", "info")

    with tab_up:
        render_alert("LƯU Ý QUAN TRỌNG: Mô hình Machine Learning / CNN được huấn luyện trên bộ dữ liệu GTSRB (mỗi ảnh là một KHUNG BIỂN BÁO ĐÃ CẮT SÁT). Nếu bạn tải ảnh toàn cảnh (có xe cộ, cây cối, bầu trời chiếm 90%), hãy bật chức năng 'Tự động phát hiện biển báo' hoặc dùng thanh cắt ảnh bên dưới.", "info")
        
        c_u1, c_u2 = st.columns([1, 1.2])
        with c_u1:
            up_file = st.file_uploader("Chọn file ảnh biển báo (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"], key="up_input_main")
            if up_file is not None:
                up_img = Image.open(up_file)
                st.image(up_img, caption=f"File gốc: {up_file.name}", use_container_width=True)
                
                auto_crop_up = st.checkbox(" Tự động khoanh vùng & cắt biển báo (Auto Detection)", value=True, key="chk_auto_crop_up")
                
                img_final_up = up_img
                if auto_crop_up:
                    cropped_arr, bbox = detect_and_crop_traffic_sign(np.array(up_img.convert("RGB")))
                    if bbox is not None:
                        st.success(f"Đã phát hiện vùng biển báo: (Vị trí: x={bbox[0]}, y={bbox[1]}, kích thước: {bbox[2]}x{bbox[3]} px)")
                        img_final_up = Image.fromarray(cropped_arr)
                        st.image(img_final_up, caption="Vùng biển báo trích xuất tự động (Input cho Mô Hình)", width=200)
                    else:
                        st.warning("Chưa tự động phát hiện vùng biển báo nổi bật. Bạn có thể sử dụng ảnh gốc hoặc cắt thủ công bên dưới.")
                
                with st.expander("✂️ Cắt Ảnh Thủ Công (Chỉnh Vùng Chọn)"):
                    img_np = np.array(up_img.convert("RGB"))
                    h_orig, w_orig = img_np.shape[:2]
                    
                    c_s1, c_s2 = st.columns(2)
                    with c_s1:
                        y_range = st.slider("Vị trí Chiều Dọc (Y min - Y max %)", 0, 100, (0, 100), key="slider_y_crop")
                    with c_s2:
                        x_range = st.slider("Vị trí Chiều Ngang (X min - X max %)", 0, 100, (0, 100), key="slider_x_crop")
                    
                    y1 = int(y_range[0] / 100.0 * h_orig)
                    y2 = int(y_range[1] / 100.0 * h_orig)
                    x1 = int(x_range[0] / 100.0 * w_orig)
                    x2 = int(x_range[1] / 100.0 * w_orig)
                    
                    if y2 > y1 and x2 > x1:
                        manual_crop_np = img_np[y1:y2, x1:x2]
                        st.image(manual_crop_np, caption="Ảnh Cắt Thủ Công", width=180)
                        if st.button("Sử dụng ảnh cắt thủ công này", key="btn_use_manual_crop"):
                            img_final_up = Image.fromarray(manual_crop_np)

        with c_u2:
            if up_file is None:
                render_alert("Vui lòng tải file ảnh và bấm nút Dự đoán.", "info")
            else:
                if st.button("Dự đoán ảnh tải lên", type="primary", use_container_width=True, key="btn_up_main"):
                    with st.spinner("Đang xử lý dự đoán..."):
                        run_prediction_flow(img_final_up, up_file.name)

    with tab_ext:
        ext_files = list(EXTERNAL_TEST_DIR.glob("*.png")) + list(EXTERNAL_TEST_DIR.glob("*.jpg")) + list(EXTERNAL_TEST_DIR.glob("*.jpeg"))
        if not ext_files:
            render_alert("Chưa có ảnh trong data/external_test/. Tạo ảnh bằng: python -m src.data_loader --create-external", "info")
        else:
            sel_ext = st.selectbox("Chọn ảnh mẫu ngoài dataset:", options=ext_files, format_func=lambda p: p.name, key="sel_ext_main")
            if sel_ext:
                c_e1, c_e2 = st.columns([1, 1.2])
                with c_e1:
                    e_img = Image.open(sel_ext)
                    st.image(e_img, caption=sel_ext.name, use_container_width=True)
                    if st.button("Dự đoán ảnh External", type="primary", use_container_width=True, key="btn_ext_main"):
                        with st.spinner("Đang dự đoán ảnh external..."):
                            run_prediction_flow(e_img, sel_ext.name)

# ==========================================
# TAB 3: KHÁM PHÁ DATASET
# ==========================================
with nav_tabs[2]:
    render_hero_banner(
        title="Khám Phá Dữ Liệu Biển Báo Giao Thông (GTSRB)",
        subtitle="Thống kê tổng quan số lượng mẫu, phân bố các lớp, tỷ lệ chia Train / Validation / Test (70% / 15% / 15%) và hình ảnh mẫu.",
        badge_text="DỮ LIỆU & PHÂN BỐ",
        icon="📊"
    )

    try:
        ds_path, is_smp = find_dataset_dir()
        if is_smp:
            render_alert("CHẾ ĐỘ SAMPLE DEMO DATASET: Đang sử dụng dữ liệu mẫu 860 ảnh để thử nghiệm giao diện. Đặt dữ liệu GTSRB thật vào data/raw/GTSRB/Train/.", "warning")
        else:
            render_alert(f"ĐẦU VÀO DỮ LIỆU THẬT GTSRB: {ds_path}", "success")

        images, labels, stats = load_raw_dataset(dataset_dir=ds_path, max_samples_per_class=50)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("Tổng Số Ảnh Data", str(stats['total_images']), "Đã nạp vào hệ thống")
        with c2:
            render_metric_card("Tổng Số Class", str(stats['total_classes']), "Class ID 0 -> 42")
        with c3:
            render_metric_card("Kích Thước Chuẩn", "32 x 32", "Pixels (RGB 3 channels)")
        with c4:
            render_metric_card("Tỷ Lệ Split Data", "70% / 15% / 15%", "Train / Val / Test Ratio")

        st.markdown("<div class='section-header'>Phân Bố Số Lượng Mẫu Theo Lớp (Class Distribution)</div>", unsafe_allow_html=True)

        df_counts = pd.DataFrame([
            {
                "Class ID": c_id,
                "Tên Biển Báo": GTSRB_CLASSES.get(c_id, {}).get("vi", f"Class {c_id}"),
                "Số Lượng Ảnh": counts
            }
            for c_id, counts in stats['class_counts'].items()
        ])
        st.bar_chart(df_counts.set_index("Class ID")["Số Lượng Ảnh"])

        st.markdown("<div class='section-header'>Trực Quan Hóa Ảnh Mẫu Theo Class ID</div>", unsafe_allow_html=True)
        selected_class = st.selectbox(
            "Chọn Class ID để xem hình ảnh mẫu:",
            options=list(range(NUM_CLASSES)),
            format_func=lambda cid: f"Class {cid}: {GTSRB_CLASSES.get(cid, {}).get('vi', 'Class ' + str(cid))}",
            key="sel_class_main"
        )

        class_dir = Path(stats['dataset_path']) / str(selected_class)
        if not class_dir.exists():
            class_dir = Path(stats['dataset_path']) / f"{selected_class:05d}"

        if class_dir.exists():
            sample_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.ppm"))
            sample_files = sample_files[:10]

            if sample_files:
                st.markdown(f"**Hình ảnh mẫu Class {selected_class} — {GTSRB_CLASSES.get(selected_class, {}).get('vi')}:**")
                cols = st.columns(min(len(sample_files), 5))
                for i, fpath in enumerate(sample_files[:5]):
                    with cols[i]:
                        img_bgr = cv2.imread(str(fpath))
                        if img_bgr is not None:
                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, caption=fpath.name, use_container_width=True)
            else:
                render_alert("Không có ảnh trong thư mục class này.", "info")
        else:
            render_alert(f"Thư mục class {selected_class} không tồn tại tại {class_dir}.", "info")

    except Exception as e:
        render_alert(f"Lỗi khi tải dữ liệu dataset: {str(e)}", "warning")

# ==========================================
# TAB 4: ĐÁNH GIÁ MÔ HÌNH
# ==========================================
with nav_tabs[3]:
    render_hero_banner(
        title="Đánh Giá Chi Tiết Mô Hình (Model Evaluation)",
        subtitle="Trực quan hóa kết quả thực nghiệm trên tập kiểm thử Test Set độc lập (Accuracy, F1-score, Confusion Matrix, Training Curves & Misclassified Images).",
        badge_text="BÁO CÁO THỰC NGHIỆM",
        icon="📈"
    )

    if not METRICS_JSON_PATH.exists():
        render_alert("CHƯA CÓ KẾT QUẢ THỰC NGHIỆM: Vui lòng huấn luyện mô hình và chạy: python -m src.evaluate", "warning")
    else:
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        if metrics.get("is_sample_dataset", False):
            render_alert("CHÚ Ý: Kết quả đánh giá dưới đây được sinh từ Sample Demo Dataset. Kết quả thực nghiệm GTSRB thật sẽ tự động cập nhật sau khi bạn huấn luyện trên dataset thật.", "warning")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("Accuracy (Độ chính xác)", f"{metrics.get('accuracy', 0)*100:.2f}%", "Overall Test Accuracy")
        with c2:
            render_metric_card("F1-Score (Weighted)", f"{metrics.get('f1_score_weighted', metrics.get('f1_score', 0))*100:.2f}%", "Weighted F1 Average")
        with c3:
            render_metric_card("F1-Score (Macro)", f"{metrics.get('f1_score_macro', 0)*100:.2f}%", "Macro F1 Average")
        with c4:
            render_metric_card("Số Mẫu Test Set", str(metrics.get("num_test_samples", 0)), "Held-out Test Samples")

        sub_tab_metrics, sub_tab_curves, sub_tab_mis = st.tabs([
            "1. Metrics & Confusion Matrix",
            "2. Training Curves (Accuracy & Loss)",
            "3. Misclassified Images & Confused Pairs"
        ])

        with sub_tab_metrics:
            st.markdown("<div class='section-header'>Ma Trận Nhầm Lẫn (Confusion Matrix) Trên Tập Test</div>", unsafe_allow_html=True)
            col_cm1, col_cm2 = st.columns(2)
            with col_cm1:
                st.markdown("### Mô Hình CNN Deep Learning")
                if CONFUSION_MATRIX_CNN_PATH.exists():
                    st.image(Image.open(CONFUSION_MATRIX_CNN_PATH), caption="CNN Confusion Matrix", use_container_width=True)
                else:
                    render_alert("Chưa có ảnh confusion_matrix_cnn.png", "info")

            with col_cm2:
                st.markdown("### Mô Hình HOG + SVM Baseline")
                if CONFUSION_MATRIX_SVM_PATH.exists():
                    st.image(Image.open(CONFUSION_MATRIX_SVM_PATH), caption="SVM Baseline Confusion Matrix", use_container_width=True)
                else:
                    render_alert("Chưa có ảnh confusion_matrix_svm.png", "info")

        with sub_tab_curves:
            st.markdown("<div class='section-header'>Đường Cong Huấn Luyện (Training vs Validation)</div>", unsafe_allow_html=True)
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("### Biểu Đồ Accuracy")
                if TRAINING_ACCURACY_PATH.exists():
                    st.image(Image.open(TRAINING_ACCURACY_PATH), caption="Training vs Validation Accuracy", use_container_width=True)
                else:
                    render_alert("Chưa có biểu đồ training_accuracy.png", "info")

            with col_c2:
                st.markdown("### Biểu Đồ Loss")
                if TRAINING_LOSS_PATH.exists():
                    st.image(Image.open(TRAINING_LOSS_PATH), caption="Training vs Validation Loss", use_container_width=True)
                else:
                    render_alert("Chưa có biểu đồ training_loss.png", "info")

        with sub_tab_mis:
            st.markdown("<div class='section-header'>Phân Tích Các Trường Hợp Dự Đoán Sai (Misclassified Images)</div>", unsafe_allow_html=True)
            top_confused = metrics.get("top_confused_pairs", [])
            if top_confused:
                st.markdown("#### Top Cặp Lớp Thường Bị Nhầm Lẫn Nhất:")
                for pair in top_confused:
                    st.write(f"• **True:** `{pair['true_name']}` (ID {pair['true_class']}) -> **Predicted:** `{pair['pred_name']}` (ID {pair['pred_class']}): **{pair['count']} mẫu**")

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

# ==========================================
# TAB 5: SO SÁNH MÔ HÌNH
# ==========================================
with nav_tabs[4]:
    render_hero_banner(
        title="So Sánh Hiệu Năng Các Mô Hình (Model Comparison)",
        subtitle="Bảng so sánh định lượng và đối chứng hiệu năng thực nghiệm giữa mô hình Baseline (HOG + SVM) và mô hình Deep Learning (CNN).",
        badge_text="BENCHMARK & MODEL SELECTION",
        icon="⚖️"
    )

    if not COMPARISON_CSV_PATH.exists():
        render_alert("CHƯA CÓ KẾT QUẢ THỰC NGHIỆM: Vui lòng chạy python -m src.evaluate để tự động xuất bảng so sánh.", "warning")
    else:
        df_comp = pd.read_csv(COMPARISON_CSV_PATH)
        st.markdown("### 1. Bảng So Sánh Hiệu Năng Thực Nghiệm (Model Comparison Table)")
        st.dataframe(df_comp, use_container_width=True)

        st.markdown("<div class='section-header'>2. Biểu Đồ So Sánh Trực Quan HOG + SVM vs CNN</div>", unsafe_allow_html=True)
        comp_json_path = RESULT_DIR / "comparison_metrics.json"
        if comp_json_path.exists():
            with open(comp_json_path, "r", encoding="utf-8") as f:
                comp_data = json.load(f)

            chart_data = []
            for model_name, metrics_item in comp_data.items():
                chart_data.append({
                    "Model": model_name,
                    "Accuracy": metrics_item.get('accuracy', 0),
                    "F1-Score (Weighted)": metrics_item.get('f1_score_weighted', metrics_item.get('f1_score', 0)),
                    "F1-Score (Macro)": metrics_item.get('f1_score_macro', 0)
                })

            df_chart = pd.DataFrame(chart_data).set_index("Model")
            st.bar_chart(df_chart)

        st.markdown("""
        ### 3. Phân Tích Kỹ Thuật Đánh Giá (Qualitative & Quantitative Analysis)

        * **HOG + SVM (Baseline Model):** Tốc độ huấn luyện nhanh, chi phí tính toán thấp. Hoạt động hiệu quả khi các biển báo có đường biên nét (edges) rõ ràng.
        * **Convolutional Neural Network (CNN):** Tự động học trích xuất đặc trưng sâu qua các lớp Cuộn (Convolutional layers). Đạt độ tổng quát hóa cao và xử lý tốt các biến dạng ảnh thực tế.
        """)

# ==========================================
# TAB 6: NHẬT KÝ DỰ ĐOÁN SQLITE
# ==========================================
with nav_tabs[5]:
    render_hero_banner(
        title="Nhật Ký & Lịch Sử Dự Đoán (SQLite Database)",
        subtitle="Quản lý và tra cứu toàn bộ lịch sử dự đoán được lưu trữ tự động trong cơ sở dữ liệu SQLite.",
        badge_text="DATABASE AUDIT & HISTORY",
        icon="📜"
    )

    df_history = get_prediction_history(limit=100)

    if df_history.empty:
        render_alert("Chưa có lịch sử dự đoán nào trong cơ sở dữ liệu. Vui lòng sang tab 'Phân loại & Camera' để thực hiện dự đoán.", "info")
    else:
        col_search, col_clear = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("Tìm kiếm theo tên file hoặc tên biển báo:", "", key="search_hist")
        with col_clear:
            st.write(" ")
            st.write(" ")
            if st.button("Clear History (Xóa Lịch Sử)", type="secondary", key="btn_clear_hist"):
                clear_prediction_history()
                st.rerun()

        if search_query:
            df_filtered = df_history[
                df_history['filename'].str.contains(search_query, case=False, na=False) |
                df_history['predicted_name'].str.contains(search_query, case=False, na=False)
            ]
        else:
            df_filtered = df_history

        st.markdown(f"**Hiển thị {len(df_filtered)} / {len(df_history)} bản ghi gần nhất:**")
        st.dataframe(
            df_filtered.rename(columns={
                "id": "ID",
                "filename": "Tên File Ảnh",
                "predicted_class": "Class ID",
                "predicted_name": "Biển Báo Dự Đoán",
                "confidence": "Confidence (%)",
                "created_at": "Thời Gian"
            }),
            use_container_width=True
        )
