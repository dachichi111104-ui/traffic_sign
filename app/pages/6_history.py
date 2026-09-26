import sys
import streamlit as st
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.history_db import get_prediction_history, clear_prediction_history
from app.components.ui_components import apply_custom_theme, render_hero_banner

st.set_page_config(page_title="Prediction History | Traffic Sign Classification", layout="wide")
apply_custom_theme()

render_hero_banner(
    title="Nhật Ký & Lịch Sử Dự Đoán (SQLite Database)",
    subtitle="Quản lý và tra cứu toàn bộ lịch sử dự đoán được lưu trữ tự động trong cơ sở dữ liệu SQLite.",
    badge_text="DATABASE AUDIT & HISTORY"
)

df_history = get_prediction_history(limit=100)

if df_history.empty:
    st.info("Chưa có lịch sử dự đoán nào trong cơ sở dữ liệu. Vui lòng qua trang 'Phân loại hình ảnh' để thực hiện dự đoán.")
else:
    col_search, col_clear = st.columns([3, 1])
    
    with col_search:
        search_query = st.text_input("Tìm kiếm theo tên file hoặc tên biển báo:", "")

    with col_clear:
        st.write(" ")
        st.write(" ")
        if st.button("Clear History (Xóa Lịch Sử)", type="secondary"):
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
