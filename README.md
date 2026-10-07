# PHÂN LOẠI VÀ NHẬN DIỆN BIỂN BÁO GIAO THÔNG (SIGANGE)

> **HỌC PHẦN:** MÁY HỌC (MACHINE LEARNING - MALE330863)  
> **ĐỀ TÀI:** XÂY DỰNG HỆ THỐNG PHÂN LOẠI VÀ NHẬN DIỆN BIỂN BÁO GIAO THÔNG ĐƯỜNG BỘ BẰNG MACHINE LEARNING VÀ DEEP LEARNING  
> **MỤC TIÊU:** Huấn luyện, đánh giá các mô hình trên ảnh biển báo và cung cấp ứng dụng Streamlit thử nghiệm dự đoán thực thời. Mặc định hướng tới bộ dữ liệu GTSRB (43 lớp), bộ nạp dữ liệu hỗ trợ cả các dataset khác.

---

## 📌 QUY TRÌNH PHÁT TRIỂN MÁY HỌC (MACHINE LEARNING PIPELINE)

Quy trình xử lý tuân thủ nghiêm ngặt chuẩn học thuật và chống rò rỉ dữ liệu (Anti Data Leakage):

$$\text{Dataset} \longrightarrow \text{Khảo sát (EDA)} \longrightarrow \text{Tiền xử lý} \longrightarrow \text{Chia tập (70/15/15)} \longrightarrow \text{Huấn luyện} \longrightarrow \text{Đánh giá} \longrightarrow \text{Demo dự đoán}$$

* **Chống rò rỉ dữ liệu (Data Leakage Protection):** Bộ dữ liệu được chia thành **Train (70%) / Validation (15%) / Test (15%)** *trước* khi áp dụng Data Augmentation.
* **Tập Train (70%):** Áp dụng Data Augmentation để tăng độ phong phú mẫu.
* **Tập Validation (15%):** Dùng để theo dõi quá trình huấn luyện, chọn checkpoint và dừng sớm (Early Stopping).
* **Tập Test (15%):** Giữ độc lập tuyệt đối để đánh giá hiệu năng cuối cùng (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).

---

## ⚙️ 10 KỸ THUẬT TIỀN XỬ LÝ & TĂNG CƯỜNG DỮ LIỆU (PREPROCESSING & AUGMENTATION)

| # | Kỹ thuật | Trạng thái | Vai trò & Tác dụng Kỹ thuật |
| :---: | :--- | :---: | :--- |
| **1** | **Resize ảnh về 32×32 (INTER_AREA)** | ✅ Đang dùng | Chuẩn hóa kích thước đầu vào đồng nhất cho CNN & HOG. |
| **2** | **CLAHE trên kênh L của LAB** | ✅ Đang dùng | Cải thiện tương phản cục bộ mà ít làm lệch màu (dành cho CNN/dự đoán CNN). |
| **3** | **Gamma Correction (Hiệu chỉnh Gamma)** | 💡 Đã tích hợp | Điều chỉnh độ sáng toàn cục phi tuyến ($\gamma > 1$ làm sáng bóng râm, $\gamma < 1$ giảm chói). |
| **4** | **Unsharp Masking (Làm sắc nét cạnh)** | 💡 Đã tích hợp | Tăng độ rõ của các đường biên và chi tiết nhỏ (số 50/80 km/h, mũi tên, biểu tượng). |
| **5** | **Chuẩn hóa pixel (/255 hoặc Z-score)** | ✅ /255 đang dùng (Z-score khả thi) | Đưa pixel về thang suitable cho gradient descent; Z-score tính mean/std từ Train. |
| **6** | **Rotation ngẫu nhiên ±12°** | ✅ Đang dùng (Augmentation) | Mô phỏng biển báo nghiêng nhẹ do góc đặt biển hoặc camera. |
| **7** | **Translation ngẫu nhiên ±3 px** | ✅ Đang dùng (Augmentation) | Mô phỏng biển báo nằm lệch khỏi tâm khung hình. |
| **8** | **Thay đổi độ sáng & Gaussian Blur** | ✅ Đang dùng (Augmentation) | Mô phỏng điều kiện ánh sáng thay đổi và ảnh mờ sương mù/thời tiết. |
| **9** | **Perspective Transform (Phối cảnh)** | 💡 Đã tích hợp (Augmentation) | Mô phỏng góc nhìn nghiêng của camera hành trình xe buýt/ô tô. |
| **10** | **Cutout / Random Erasing** | 💡 Đã tích hợp (Augmentation) | Xóa ngẫu nhiên vùng ảnh, xử lý vấn đề biển báo bị vật thể (lá cây, cột) che khuất. |
| **+** | **Motion Blur (Mờ chuyển động)** | 💡 Đã tích hợp (Augmentation) | Mô phỏng hiện tượng mờ ảnh theo góc chuyển động khi xe chạy tốc độ cao. |

### Chi Tiết Kỹ Thuật Nâng Cao:
* **Gamma Correction:** Tăng cường hoặc giảm bớt mức độ chiếu sáng phi tuyến tính.
* **Unsharp Masking:** Làm rõ nét các thông số số hiệu tốc độ, biểu tượng chỉ hướng trên biển báo.
* **Perspective Transform:** Tạo ra các góc nghiêng phối cảnh thực tế của phương tiện giao thông.
* **Cutout / Random Erasing:** Buộc mô hình cuộn học toàn bộ cấu trúc biển báo thay vì chỉ phụ thuộc vào một chi tiết đơn lẻ.
* **Per-Channel Z-score Normalization:** Chuẩn hóa $X_{norm} = \frac{X - \mu}{\sigma + 1e-7}$ giúp CNN hội tụ nhanh và ổn định hơn.

---

## ⚖️ KỸ THUẬT XỬ LÝ DỮ LIỆU MẤT CÂN BẰNG (IMBALANCED DATA SAMPLING)

Dự án tích hợp các phương pháp cân bằng mẫu cho tập dữ liệu imbalanced:
1. **Random Undersampling:** Lấy mẫu ngẫu nhiên giảm số lượng của các lớp đa số về số mẫu mục tiêu.
2. **NearMiss (NearMiss-1):** Giữ lại các mẫu thuộc lớp đa số có khoảng cách trung bình nhỏ nhất tới K mẫu thuộc lớp thiểu số kề cận (dùng K-NN).
3. **Cluster Centroids:** Sử dụng thuật toán **K-Means** trên lớp đa số để thay thế các tập mẫu bằng các tâm cụm đại diện.
4. **Class Weighting:** Tính trọng số mất mát $w_c = \frac{N}{K \cdot N_c}$ hỗ trợ huấn luyện CNN trực tiếp mà không làm mất mẫu dữ liệu.

---

## 🤖 MÔ HÌNH VÀ ĐÁNH GIÁ (MODELS & EVALUATION)

* **HOG + SVM (RBF Kernel):** Mô hình Machine Learning baseline. Trích xuất 324 chiều đặc trưng **HOG (Histogram of Oriented Gradients)** kết hợp chuẩn hóa **StandardScaler** và phân loại bằng RBF SVM.
* **Custom CNN:** Mạng tích chập 3 khối Conv2D với **Batch Normalization**, **Dropout (0.25/0.5)**, huấn luyện kèm **Class Weighting**, **Early Stopping** và **ReduceLROnPlateau**.
* **MobileNetV2:** Tùy chọn Transfer Learning tiên tiến tích hợp sẵn trong mã nguồn.
* **Chỉ số đánh giá:** Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted), Ma trận nhầm lẫn (Confusion Matrix) và phân tích ảnh dự đoán sai.

---

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT STRUCTURE)

```text
traffic_sign/
├── app/                        # Giao diện Web App Streamlit (Dashboard, Phân loại, Dataset, Đánh giá, So sánh, Lịch sử)
│   ├── app.py                  # File chính ứng dụng Web Streamlit
│   └── components/             # Các thành phần UI (Header, Hero, Cards, Alerts)
├── data/                       # Thư mục dữ liệu gốc (GTSRB), dữ liệu mẫu (sample), external test & lịch sử SQLite
├── models/                     # Mô hình đã huấn luyện (.keras, .pkl), class mapping & metadata
├── notebooks/                  # 5 Notebooks Jupyter chuẩn học thuật (EDA, Preprocessing, SVM, CNN, Evaluation)
├── results/                    # Kết quả thực nghiệm (Metrics JSON, Comparison CSV, Confusion Matrix, Training Curves)
├── src/                        # Các mô-đun Python (data_loader, config, preprocessing, feature_extraction, train_svm, train_cnn, evaluate, predict)
├── requirements.txt            # Thư viện phụ thuộc
└── README.md                   # Tài liệu chi tiết đồ án
```

---

## 🚀 CÀI ĐẶT VÀ CHẠY DỰ ÁN (SETUP & EXECUTION)

Chạy các lệnh tại thư mục gốc `traffic_sign/`:

### 1. Khởi tạo môi trường và cài đặt thư viện
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Tạo dữ liệu mẫu (nếu chưa có dataset thật)
```bash
python -m src.data_loader --create-sample
```

### 3. Huấn luyện mô hình và khởi chạy Web App
```bash
# 1. Huấn luyện mô hình Baseline HOG + SVM
python -m src.train_svm

# 2. Huấn luyện mô hình Deep Learning Custom CNN
python -m src.train_cnn

# (Tùy chọn) Sử dụng cờ --retrain để huấn luyện lại CNN khi file model đã tồn tại:
python -m src.train_cnn --retrain

# 3. Khởi chạy ứng dụng Web Streamlit
streamlit run app/app.py
```

---

## 👨‍💻 THÔNG TIN HỌC PHẦN
- **Học phần**: Máy học (Machine Learning)
- **Khoa**: Công nghệ Thông tin
- **Năm học**: 2025 - 2026
