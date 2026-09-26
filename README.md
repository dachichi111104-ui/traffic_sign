# TRAFFIC SIGN IMAGE CLASSIFICATION & RECOGNITION (SIGANGE)

> **HỌC PHẦN:** MÁY HỌC (MACHINE LEARNING - MALE330863)  
> **ĐỀ TÀI:** XÂY DỰNG HỆ THỐNG PHÂN LOẠI BIỂN BÁO GIAO THÔNG ĐƯỜNG BỘ DỰA TRÊN MACHINE LEARNING & DEEP LEARNING

---

## TỔNG QUAN ĐỒ ÁN (PROJECT OVERVIEW)

Dự án **SIGANGE** là một hệ thống phân loại và nhận diện biển báo giao thông đường bộ tự động toàn diện từ A-Z, tuân thủ nghiêm ngặt quy trình phát triển Machine Learning chuẩn học thuật:

$$\text{DATASET} \longrightarrow \text{EDA} \longrightarrow \text{PREPROCESSING} \longrightarrow \text{FEATURE EXTRACTION} \longrightarrow \text{MODEL TRAINING} \longrightarrow \text{EVALUATION} \longrightarrow \text{WEB APP DEMO}$$

### Các Điểm Nổi Bật Kỹ Thuật (Key Features):
1. **Tiền Xử Lý Nâng Cao (CLAHE)**: Cân bằng tương phản thích ứng thích nghi trên không gian màu **LAB** (kênh $L$), giúp nhận diện tốt biển báo bị bóng râm, thiếu sáng hoặc chói nắng.
2. **Chống Rò Rỉ Dữ Liệu (Anti Data Leakage)**: Phân chia tập dữ liệu **Train (70%) / Validation (15%) / Test (15%)** *trước* khi tăng cường dữ liệu (Data Augmentation).
3. **Mô Hình Đối Chứng (Baseline & Deep Learning)**:
   - **Baseline ML**: Trích xuất đặc trưng **HOG (Histogram of Oriented Gradients)** 324 chiều kết hợp chuẩn hóa **StandardScaler** và phân loại bằng **RBF SVM** ($\approx 96.96\%$ Accuracy).
   - **Deep Learning CNN**: Kiến trúc mạng cuộn 3 khối Conv2D với **BatchNormalization**, **Dropout (0.25/0.5)**, **ReduceLROnPlateau**, **Class Weighting** và **EarlyStopping** ($\approx 98.5\%+$ Accuracy).
   - **Transfer Learning**: Tích hợp mô hình pretrained **MobileNetV2** cho bài toán phân loại nâng cao.
4. **Giao Diện Web App Sang Trọng**: Giao diện chuẩn doanh nghiệp / học thuật (phong cách Royal Navy `#0F172A`), hỗ trợ **Bật/Tắt Webcam**, **Tự động khoanh vùng & cắt biển báo (Smart Auto Sign Crop)** và **Cắt ảnh thủ công (Manual Crop Sliders)**.

---

## CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT STRUCTURE)

```text
sigange/
├── app/                        # Giao diện Web App Streamlit
│   ├── app.py                  # File chạy chính ứng dụng Streamlit (Top Header Nav)
│   └── components/             # Các thành phần UI sang trọng (Header, Hero, Metric Cards)
├── data/                       # Thư mục dữ liệu (Đã được gitignore để giữ repo nhẹ)
│   ├── raw/                    # Chứa bộ dữ liệu gốc GTSRB (Train/0..42)
│   ├── sample/                 # Bộ dữ liệu mẫu nhẹ để chạy thử nghiệm pipeline
│   └── external_test/          # Ảnh ngoài dataset phục vụ kiểm thử
├── models/                     # Thư mục chứa model đã huấn luyện (.keras, .pkl)
├── notebooks/                  # 5 Notebooks Jupyter chuẩn học thuật
│   ├── 01_eda.ipynb            # Khảo sát dữ liệu & phân bố lớp
│   ├── 02_preprocessing.ipynb  # Tiền xử lý CLAHE, Normalization & Data Augmentation
│   ├── 03_svm_baseline.ipynb   # Trích xuất đặc trưng HOG & Huấn luyện SVM
│   ├── 04_cnn_training.ipynb   # Huấn luyện CNN, BatchNormalization & Transfer Learning
│   └── 05_evaluation.ipynb     # Đánh giá Test Set, Confusion Matrix & Error Analysis
├── results/                    # Kết quả thực nghiệm (Conf Matrix, Model Comparison CSV)
├── src/                        # Mã nguồn Python mô-đun hóa
│   ├── config.py               # Cấu hình siêu tham số, đường dẫn & 43 lớp GTSRB
│   ├── data_loader.py          # Nạp dữ liệu tự động (GTSRB thật hoặc Sample Demo)
│   ├── preprocessing.py        # Pipeline CLAHE, Split, Augmentation & Auto-Crop
│   ├── feature_extraction.py   # Trích xuất đặc trưng HOG
│   ├── train_svm.py            # Huấn luyện mô hình HOG + SVM
│   ├── train_cnn.py            # Huấn luyện mô hình Custom CNN & MobileNetV2
│   ├── evaluate.py             # Đánh giá tập Test độc lập & trích xuất ảnh sai
│   └── predict.py              # Pipeline dự đoán cho ứng dụng Web
├── .gitignore                  # Bỏ qua dữ liệu nặng và file model tạm
├── requirements.txt            # Danh sách thư viện Python cần thiết
└── README.md                   # Tài liệu hướng dẫn đồ án
```

---

## HƯỚNG DẪN CÀI ĐẶT & CHẠY DEMO CHO THÀNH VIÊN NHÓM

### 1. Cài đặt môi trường
Mở Terminal / PowerShell tại thư mục dự án và chạy các lệnh:

```bash
# Tạo môi trường ảo Python (Khuyên dùng)
python -m venv .venv

# Kích hoạt môi trường ảo (Windows)
.venv\Scripts\activate

# Cài đặt toàn bộ thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Chạy Ứng Dụng Web Streamlit (Chế Độ Demo Nhanh)
Bạn **không cần tải bộ dữ liệu nặng** ngay lập tức. Hệ thống có sẵn chế độ **Sample Demo** tự động giúp thành viên nhóm chạy được ngay ứng dụng để xem giao diện và thử nghiệm:

```bash
# Tạo dữ liệu mẫu nhẹ (nếu chưa có)
python -m src.data_loader --create-sample

# Chạy ứng dụng Web
streamlit run app/app.py
```
Mở trình duyệt tại đường dẫn `http://localhost:8501`.

---

## KẾT QUẢ THỰC NGHIỆM ĐÁNH GIÁ MÔ HÌNH (EXPERIMENTAL RESULTS)

Đánh giá độc lập trên tập **Test Set (15% - 5,882 ảnh)**:

| Mô Hình (Model) | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HOG + RBF SVM (Baseline)** | **96.96%** | 97.85% | 97.71% | 97.78% | 96.95% |
| **Custom CNN (Deep Learning)** | **98.65%** | 98.92% | 98.80% | 98.86% | 98.64% |

---

## HƯỚNG DẪN PUSH LÊN GITHUB & DEPLOY ONLINE MIỄN PHÍ

### Bước 1: Đẩy mã nguồn lên GitHub (Dung lượng siêu nhẹ < 5MB)
File `.gitignore` đã được cấu hình sẵn để **bỏ qua bộ dữ liệu nặng và các file model lớn**. Bạn chỉ cần đẩy phần mã nguồn nhẹ lên GitHub:

```bash
# 1. Khởi tạo Git repo (nếu chưa có)
git init

# 2. Thêm tất cả file mã nguồn
git add .

# 3. Commit
git commit -m "Feat: Complete Traffic Sign Classification Pipeline & Streamlit Web App"

# 4. Liên kết với GitHub Repo của bạn và Push
git remote add origin https://github.com/USERNAME/sigange.git
git branch -M main
git push -u origin main
```

### Bước 2: Deploy ứng dụng Web chạy Online Miễn Phí (Streamlit Community Cloud)
Để tất cả thành viên trong nhóm và Thầy/Cô có thể mở liên kết web dùng thử trực tiếp mà không cần cài đặt code:

1. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub.
2. Bấm nút **New app**.
3. Chọn Repository `sigange`, Branch `main`, và Main file path: `app/app.py`.
4. Bấm **Deploy!** 
5. Bạn sẽ nhận được một đường link public (Ví dụ: `https://sigange-traffic-sign.streamlit.app`) để chia sẻ cho cả nhóm và đưa vào báo cáo môn học!

---

## TÁC GIẢ & THÔNG TIN HỌC PHẦN
- **Đồ án môn học**: Máy học (Machine Learning)
- **Khoa**: Công nghệ Thông tin
- **Năm học**: 2025 - 2026
