import sys
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import EXTERNAL_TEST_DIR
from src.predict import predict_traffic_sign
from src.history_db import add_prediction_to_history
from app.components.ui_components import apply_custom_theme, render_hero_banner, render_alert

st.set_page_config(page_title="Phân Loại & Camera | Traffic Sign Classification", layout="wide")
apply_custom_theme()

render_hero_banner(
    title="Nhận Diện & Phân Loại Biển Báo Giao Thông",
    subtitle="Hỗ trợ 3 phương thức đầu vào: Chụp ảnh trực tiếp từ Camera, Tải file ảnh lên (Upload) hoặc Thử nghiệm bộ ảnh ngoài dataset.",
    badge_text="MÔ ĐUN DỰ ĐOÁN THỰC THỜI"
)

selected_model = st.radio(
    "Mô hình dự đoán:",
    ["CNN", "HOG + SVM"],
    index=0,
    horizontal=True
)

tab_camera, tab_upload, tab_external = st.tabs([
    "1. Chụp Hình Trực Tiếp (Camera / Webcam)",
    "2. Tải Ảnh Lên (Upload File)",
    "3. Ảnh Thử Nghiệm Ngoài Dataset (External Test)"
])

def process_and_render_prediction(img_input, filename_label):
    try:
        if isinstance(img_input, Image.Image):
            img_array = np.array(img_input.convert("RGB"))
        else:
            img_array = img_input

        result = predict_traffic_sign(img_array, model_type=selected_model)

        render_alert(f"Dự đoán hoàn tất thành công sử dụng mô hình {selected_model}.", "success")

        st.markdown(f"""
            <div class="result-card">
                <div class="result-title">{result['name_vi']}</div>
                <div class="result-detail"><b>Tên tiếng Anh:</b> {result['name_en']}</div>
                <div class="result-detail"><b>Class ID:</b> {result['class_id']}</div>
                <div class="result-detail"><b>Độ tin cậy (Confidence):</b> <span style="color:#2563EB; font-weight:800;">{result['confidence']:.2f}%</span></div>
            </div>
        """, unsafe_allow_html=True)

        st.progress(min(result['confidence'] / 100.0, 1.0))

        st.markdown("#### Top 3 Dự Đoán Có Xác Suất Cao Nhất:")
        for idx, pred in enumerate(result['top_3'], 1):
            c1, c2, c3 = st.columns([0.5, 3.5, 1.5])
            with c1:
                st.write(f"**#{idx}**")
            with c2:
                st.write(f"**{pred['name_vi']}** *(ID: {pred['class_id']})*")
            with c3:
                st.write(f"**{pred['confidence']:.2f}%**")

        # Save record to SQLite
        try:
            add_prediction_to_history(
                filename=filename_label,
                predicted_class=result['class_id'],
                predicted_name=result['name_vi'],
                confidence=result['confidence']
            )
        except Exception:
            pass

    except Exception as e:
        render_alert(f"Lỗi khi thực hiện dự đoán: {str(e)}", "warning")


# --- TAB 1: CAMERA CAPTURE ---
with tab_camera:
    col_cam1, col_cam2 = st.columns([1, 1.2])

    with col_cam1:
        camera_file = st.camera_input("Chụp ảnh biển báo trước ống kính...")
        
    with col_cam2:
        if camera_file is not None:
            cam_img = Image.open(camera_file)
            btn_cam = st.button("Dự đoán ảnh từ Camera", type="primary", use_container_width=True, key="btn_cam")
            if btn_cam:
                with st.spinner("Đang phân tích hình ảnh từ Camera..."):
                    process_and_render_prediction(cam_img, "Camera_Snapshot.jpg")
        else:
            render_alert("Bấm nút chụp hình trên camera để tải ảnh lên và nhận diện.", "info")


# --- TAB 2: UPLOAD FILE ---
with tab_upload:
    col_up1, col_up2 = st.columns([1, 1.2])

    with col_up1:
        uploaded_file = st.file_uploader("Chọn file ảnh biển báo (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"], key="file_up")
        if uploaded_file is not None:
            up_image = Image.open(uploaded_file)
            st.image(up_image, caption=f"File: {uploaded_file.name}", use_container_width=True)

    with col_up2:
        if uploaded_file is None:
            render_alert("Vui lòng tải file ảnh và bấm nút Dự đoán.", "info")
        else:
            btn_up = st.button("Dự đoán ảnh tải lên", type="primary", use_container_width=True, key="btn_up")
            if btn_up:
                with st.spinner("Đang chạy dự đoán..."):
                    process_and_render_prediction(up_image, uploaded_file.name)


# --- TAB 3: EXTERNAL TEST ---
with tab_external:
    ext_files = list(EXTERNAL_TEST_DIR.glob("*.png")) + list(EXTERNAL_TEST_DIR.glob("*.jpg")) + list(EXTERNAL_TEST_DIR.glob("*.jpeg"))

    if not ext_files:
        render_alert("Chưa có ảnh trong data/external_test/. Tạo ảnh bằng: python -m src.data_loader --create-external", "info")
    else:
        selected_ext_file = st.selectbox("Chọn ảnh mẫu ngoài dataset:", options=ext_files, format_func=lambda p: p.name)
        
        if selected_ext_file:
            col_ext1, col_ext2 = st.columns([1, 1.2])
            with col_ext1:
                ext_img = Image.open(selected_ext_file)
                st.image(ext_img, caption=selected_ext_file.name, use_container_width=True)
                btn_ext = st.button("Dự đoán ảnh External", type="primary", use_container_width=True, key="btn_ext")

            with col_ext2:
                if btn_ext:
                    with st.spinner("Đang dự đoán ảnh external..."):
                        process_and_render_prediction(ext_img, selected_ext_file.name)
