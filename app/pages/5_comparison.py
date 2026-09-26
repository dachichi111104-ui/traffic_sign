import sys
import json
import pandas as pd
import streamlit as st
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import RESULT_DIR, COMPARISON_CSV_PATH
from app.components.ui_components import apply_custom_theme, render_hero_banner

st.set_page_config(page_title="Model Comparison | Traffic Sign Classification", layout="wide")
apply_custom_theme()

render_hero_banner(
    title="So Sánh Hiệu Năng Các Mô Hình (Model Comparison)",
    subtitle="Bảng so sánh định lượng và đối chứng hiệu năng thực nghiệm giữa mô hình Baseline (HOG + SVM) và mô hình Deep Learning (CNN).",
    badge_text="BENCHMARK & MODEL SELECTION"
)

if not COMPARISON_CSV_PATH.exists():
    st.warning(
        "⚠️ **CHƯA CÓ KẾT QUẢ THỰC NGHIỆM:**\n"
        "Khái niệm số liệu chưa tồn tại tại `results/model_comparison.csv`.\n"
        "Vui lòng thực hiện đánh giá để tự động xuất bảng so sánh:\n"
        "`python -m src.evaluate`"
    )
    st.stop()

# Load real comparison CSV
df_comp = pd.read_csv(COMPARISON_CSV_PATH)

st.markdown("### 1. Bảng So Sánh Hiệu Năng Thực Nghiệm (Model Comparison Table)")
st.dataframe(df_comp, use_container_width=True)

st.markdown("<div class='section-header'>2. Biểu Đồ So Sánh Trực Quan HOG + SVM vs CNN</div>", unsafe_allow_html=True)

comp_json_path = RESULT_DIR / "comparison_metrics.json"
if comp_json_path.exists():
    with open(comp_json_path, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    chart_data = []
    for model_name, metrics in comp_data.items():
        chart_data.append({
            "Model": model_name,
            "Accuracy": metrics.get('accuracy', 0),
            "F1-Score (Weighted)": metrics.get('f1_score_weighted', metrics.get('f1_score', 0)),
            "F1-Score (Macro)": metrics.get('f1_score_macro', 0)
        })

    df_chart = pd.DataFrame(chart_data).set_index("Model")
    st.bar_chart(df_chart)

st.markdown("""
### 3. Phân Tích Kỹ Thuật Đánh Giá (Qualitative & Quantitative Analysis)

* **HOG + SVM (Baseline Model):**
  - **Ưu điểm:** Tốc độ huấn luyện nhanh, chi phí tính toán thấp. Hoạt động cực kỳ hiệu quả khi các biển báo có đường biên nét (edges/contours) rõ ràng và ít bị biến dạng.
  - **Hạn chế:** Phụ thuộc vào việc trích xuất đặc trưng thủ công (hand-crafted features). Nhạy cảm với ánh sáng biến đổi mạnh, góc nghiêng hoặc biển báo bị che khuất một phần.

* **Convolutional Neural Network (CNN - Deep Learning):**
  - **Ưu điểm:** Tự động học biểu diễn đặc trưng cấp thấp (cạnh, đường nét) đến cấp cao (hình dạng biển báo, biểu tượng chữ/số/mũi tên). Khả năng tổng quát hóa (generalization) cao hơn khi gặp các điều kiện ánh sáng ngẫu nhiên, nhiễu hoặc biển báo hơi mờ.
  - **Hạn chế:** Cần lượng dữ liệu lớn và nhiều epoch huấn luyện để đạt được trọng số hội tụ tối ưu.

* **Kết luận Đồ án:**
  Mô hình CNN được lựa chọn là mô hình dự đoán chính cho hệ thống phân loại biển báo giao thông tự động, với HOG + SVM đóng vai trò là cột mốc so sánh (Benchmark Baseline).
""")
