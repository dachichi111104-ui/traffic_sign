import sys
import json
import streamlit as st
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import RESULT_DIR, MODEL_DIR, MODEL_METADATA_PATH, CNN_MODEL_PATH, SVM_MODEL_PATH, NUM_CLASSES
from src.data_loader import find_dataset_dir
from app.components.ui_components import apply_custom_theme, render_hero_banner, render_metric_card

st.set_page_config(page_title="Dashboard | Traffic Sign Classification", layout="wide")
apply_custom_theme()

render_hero_banner(
    title="Tổng Quan Hệ Thống & Trạng Thái Mô Hình",
    subtitle="Bảng điều khiển tổng hợp thông số dữ liệu, tình trạng huấn luyện của các mô hình HOG + SVM và CNN TensorFlow.",
    badge_text="DASHBOARD TRUNG TÂM"
)

# Check Dataset & Model Status
dataset_path, is_sample = find_dataset_dir()

if is_sample:
    st.warning(
        "⚠️ **CHẾ ĐỘ KIỂM THỬ PIPELINE (SAMPLE DEMO DATASET):**\n"
        "Hệ thống hiện tại chưa phát hiện bộ dữ liệu GTSRB thật trong `data/raw/GTSRB/Train/`.\n"
        "Hệ thống đang chạy trên **Sample Dataset** để kiểm thử giao diện và pipeline.\n"
        "*(Số liệu thực nghiệm chính thức sẽ tự động cập nhật khi bạn đặt dữ liệu GTSRB thật và thực hiện huấn luyện)*."
    )
elif dataset_path is not None:
    st.success(f"✅ **ĐÃ KẾT NỐI BỘ DỮ LIỆU GTSRB THẬT:** Dữ liệu được nạp từ `{dataset_path}`")
else:
    st.error("❌ **CHƯA CÓ DỮ LIỆU:** Vui lòng đặt dữ liệu vào `data/raw/GTSRB/Train/` hoặc chạy `python -m src.data_loader --create-sample` để tạo sample demo.")

# Read metrics from actual results
cnn_acc = "Chưa train"
svm_acc = "Chưa train"

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

# Render Top Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Bộ dữ liệu", "GTSRB" if not is_sample else "Sample Demo", "German Traffic Sign")
with c2:
    render_metric_card("Số lượng Lớp", str(NUM_CLASSES), "Classes (0 - 42)")
with c3:
    render_metric_card("Độ chính xác CNN", cnn_acc, "Deep Learning Model")
with c4:
    render_metric_card("Baseline HOG + SVM", svm_acc, "Machine Learning Baseline")

st.markdown("<div class='section-header'>Quy Trình Xử Lý Machine Learning Pipeline</div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("""
    ### Các Bước Xử Lý Chuẩn Học Máy (End-to-End Pipeline)
    1. **Data Loading & Preprocessing:** Tải hình ảnh, chuẩn hóa không gian màu RGB, resize về $32 \\times 32$ pixel, chuẩn hóa pixel về dải $[0, 1]$.
    2. **Strict Train / Val / Test Split (70% / 15% / 15%):** Phân chia tập dữ liệu chống rò rỉ (Data Leakage). Áp dụng xoay ($\pm 12^\circ$), dịch chuyển, đổi độ sáng *duy nhất trên tập Training*.
    3. **Baseline Model (HOG + SVM):** Trích xuất đặc trưng HOG (Histogram of Oriented Gradients) và phân loại bằng SVM nhân RBF.
    4. **Deep Learning Model (CNN):** Mạng cuộn Conv2D 3 khối Keras TensorFlow với Dropout, EarlyStopping, ModelCheckpoint và ReduceLROnPlateau.
    5. **Evaluation & Comparison:** Đánh giá trên tập Test độc lập, xuất Confusion Matrix, Classification Report, phân tích ảnh dự đoán sai (Misclassified Images) và so sánh trực quan.
    """)

with col_right:
    st.markdown("""
    ### Trạng Thái Mô Hình Hiện Tại
    """)
    if CNN_MODEL_PATH.exists():
        st.success("✅ **Model CNN:** Đã huấn luyện (`models/cnn_traffic_sign.keras`)")
    else:
        st.warning("⚠️ **Model CNN:** Chưa tìm thấy file mô hình.")

    if SVM_MODEL_PATH.exists():
        st.success("✅ **Model HOG + SVM Baseline:** Đã huấn luyện (`models/svm_hog_model.pkl`)")
    else:
        st.warning("⚠️ **Model HOG + SVM:** Chưa tìm thấy file mô hình.")

    st.markdown("""
    ### Chức Năng Hệ Thống
    * **Phân loại & Camera:** Chụp hình trực tiếp từ Webcam, upload ảnh hoặc test ảnh ngoài dataset (External Image Test), xem kết quả Top 3 confidence.
    * **Dataset Explorer:** Thống kê phân bố lớp, phân chia tỷ lệ 70/15/15 và xem ảnh mẫu từng nhóm biển báo.
    * **Model Evaluation:** Đồ thị Loss/Accuracy, Confusion Matrix thực tế và xem các ảnh bị dự đoán sai.
    * **Model Comparison:** So sánh đối chứng trực quan HOG + SVM vs CNN từ file CSV thực nghiệm.
    * **Prediction History:** Lưu vết nhật ký dự đoán tự động vào SQLite.
    """)
