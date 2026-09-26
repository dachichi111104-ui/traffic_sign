import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import GTSRB_CLASSES, NUM_CLASSES, RAW_DATA_DIR, SAMPLE_DATA_DIR, RESULT_DIR
from src.data_loader import load_raw_dataset, find_dataset_dir
from app.components.ui_components import apply_custom_theme, render_hero_banner, render_metric_card, render_alert

st.set_page_config(page_title="Dataset Explorer | Traffic Sign Classification", layout="wide")
apply_custom_theme()

render_hero_banner(
    title="Khám Phá Dữ Liệu Biển Báo Giao Thông (GTSRB)",
    subtitle="Thống kê tổng quan số lượng mẫu, phân bố các lớp, tỷ lệ chia Train / Validation / Test (70% / 15% / 15%) và hình ảnh mẫu.",
    badge_text="DỮ LIỆU & PHÂN BỐ"
)

try:
    dataset_path, is_sample = find_dataset_dir()
    
    if is_sample:
        render_alert("CHẾ ĐỘ SAMPLE DEMO DATASET: Đang sử dụng dữ liệu mẫu 860 ảnh để thử nghiệm giao diện. Đặt dữ liệu GTSRB thật vào data/raw/GTSRB/Train/.", "warning")
    else:
        render_alert(f"ĐẦU VÀO DỮ LIỆU THẬT GTSRB: {dataset_path}", "success")

    images, labels, stats = load_raw_dataset(dataset_dir=dataset_path, max_samples_per_class=50)

    # Top stats metrics
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

    counts_dict = stats['class_counts']
    df_counts = pd.DataFrame([
        {
            "Class ID": c_id,
            "Tên Biển Báo": GTSRB_CLASSES.get(c_id, {}).get("vi", f"Class {c_id}"),
            "Số Lượng Ảnh": counts
        }
        for c_id, counts in counts_dict.items()
    ])

    st.bar_chart(df_counts.set_index("Class ID")["Số Lượng Ảnh"])

    st.markdown("<div class='section-header'>Trực Quan Hóa Ảnh Mẫu Theo Class ID</div>", unsafe_allow_html=True)

    selected_class = st.selectbox(
        "Chọn Class ID để xem hình ảnh mẫu:",
        options=list(range(NUM_CLASSES)),
        format_func=lambda cid: f"Class {cid}: {GTSRB_CLASSES.get(cid, {}).get('vi', 'Class ' + str(cid))}"
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
            
            if len(sample_files) > 5:
                cols2 = st.columns(min(len(sample_files) - 5, 5))
                for i, fpath in enumerate(sample_files[5:10]):
                    with cols2[i]:
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
