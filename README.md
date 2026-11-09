# HE THONG PHAN LOAI VA NHAN DIEN BIEN BAO GIAO THONG (SIGANGE)
## Comprehensive Traffic Sign Recognition: Classical Machine Learning & Deep Learning Benchmarks

**Hoc phan:** May hoc (Machine Learning - MALE330863)  
**De tai:** Xay dung he thong phan loai va nhan dien bien bao giao thong duong bo bang Machine Learning va Deep Learning  
**Giang vien huong dan:** TS. Tran Hoang Loc  
**Tap du lieu chuan:** German Traffic Sign Recognition Benchmark (GTSRB - 43 danh muc bien bao)  

---

## MUC LUC TAI LIEU
1. Tong quan He thong & Tinh Cap thiet
2. Pipeline May hoc Chuyen nghiep & Giao thuc Chong ro ri Du lieu (Anti-Leakage)
3. Quy trinh Tien xu ly & Bo tang cuong Du lieu Ngoai canh (Data Augmentation 2.0)
4. Huong dan Chi tiet Tu Huan luyen Lai Cac Mo hinh (Model Retraining Guide)
5. Danh gia Hieu nang & Do luong Do ben vung (Robustness Stress Testing)
6. Kien truc Ung dung Web Streamlit & Co so Du lieu SQLite
7. Cau truc Thu muc Du an (Directory Structure)

---

## 1. TONG QUAN HE THONG & TINH CAP THIET

He thong nhan dien bien bao giao thong la thanh phan cot loi trong he thong ho tro lai xe nang cao (ADAS) va xe tu hanh. Trong moi truong van hanh thuc te, camera thuong xuyen phai doi mat voi 6 dieu kien ngoai canh khac nghiet:
* **Mo chuyen dong (Motion Blur):** Do phuong tien di chuyen o van toc cao hoac mat duong rung lac.
* **Mo mat net & Suong mu (Gaussian / Defocus Blur):** Do thoi tiet am uot, mua phuon hoac ong kinh camera bam bui.
* **Thieu sang & Bong ram (Low-Light / Tunnel Entry):** Luc hoang hon, troi toi hoac khi di xuyen qua ham duong bo.
* **Choi loa & Thua sang (Overexposure / Sunlight Glare):** Anh nang mat troi chieu truc dien gay bao hoa cam bien.
* **Nhieu hat cam bien (Sensor Noise / High ISO):** Cac thiet bi dashcam gia re hoac chup thieu sang gay hat nhieu dot.
* **Che khuat vat ly mot phan (Occlusion / Cutout):** Bien bao bi che boi canh cay, cot den hoac bui ban.

Du an nay xay dung mot Machine Learning Pipeline hoan chinh, tich hop cac co che tien xu ly thich ung va tang cuong du lieu chuyen sau giup mo hinh phan biet chinh xac cac chu so de gay nham lan (dac biet la bien **50 km/h** va **80 km/h**) dong thoi duy tri do ben vung cao duoi cac dieu kien ngoai canh tren.

---

## 2. PIPELINE CHONG RO RI DU LIEU (ANTI DATA LEAKAGE PROTOCOL)

Quy trinh phat trien duoc chuan hoa theo quy trinh 6 buoc nghiem ngat:

$$\text{Raw Dataset} \xrightarrow{\text{Stratified Split}} \begin{cases} \text{Train (70%)} \xrightarrow{\text{Augmentation 2.0}} \text{Huan luyen mo hinh} \\ \text{Val (15%)} \xrightarrow{\text{Giuyen ban}} \text{Early Stopping / Checkpoint} \\ \text{Test (15%)} \xrightarrow{\text{Giu nguyen ban}} \text{Danh gia doc lap} \end{cases}$$

### Nguyen tac bat bien:
* **Stratified Train / Val / Test Split (70% / 15% / 15%):** Thuc hien phan chia phan tang ngay tu anh tho (Raw Input) truoc khi ap dung bat ky phep bien doi nao de bao toan nguyen ven ty le giua 43 lop.
* **Tuyet doi khong Augmentation tren Validation va Test:** Tang cuong du lieu (Rotation, Translation, Blurring, Noise, Cutout) chi duoc phep ap dung tren tap Train (70%). Tap Val va Test giu nguyen ban 100% nham phan anh chinh xac kha nang tong quat hoa.
* **Co lap tham so chuan hoa:** Cac he so thong ke (Gia tri Mean, Std neu su dung Z-Score) chi duoc tinh toan doc quyen tren tap Train va ap dung dong nhat sang Val va Test.

---

## 3. TIEN XU LY VA TANG CUONG DU LIEU (DATA AUGMENTATION 2.0)

### A. Bang tong hop 10 ky thuat xu ly anh

| STT | Ky thuat | Pham vi thuc thi | Muc tieu & Vai tro ky thuat |
| :---: | :--- | :---: | :--- |
| **01** | **Resize 32x32 (cv2.INTER_AREA)** | Train / Val / Test | Bao toan toi da net vien va ky tu so so voi phep song tuyen tinh. |
| **02** | **CLAHE tren kenh L (LAB Color)** | Tien xu ly thich ung | Nang cao do tuong phan vung toi ma khong lam sai lech mau nhan dang (do/xanh/vang). |
| **03** | **Gamma Correction phi tuyen** | Tien xu ly thich ung | Chinh do sang ($\gamma = 1.2$ lam sang bong ram, $\gamma = 0.8$ giam chao sang). |
| **04** | **Unsharp Masking (addWeighted)** | Tien xu ly thich ung | Lam sac net cac bien ky tu so 30, 50, 80 va ky hieu bien bao. |
| **05** | **Loc nhieu Trung vi (Median Blur 3x3)** | Tien xu ly thich ung | Xoa sach nhieu hat sensor noise roi vao khoang ho chu so 5. |
| **06** | **Xoay ngau nhien (-12° den +12°)** | Train Augmentation | Mo phong goc nghieng do camera dat lech tam (KHONG lat ngang). |
| **07** | **Dich chuyen tinh tien (Shift ±3 px)** | Train Augmentation | Mo phong bien bao nam lech tam khung hinh. |
| **08** | **Perspective Transform (Warping)** | Train Augmentation | Mo phong goc nhin xe chay nghiem tu kinh chan gio. |
| **09** | **Mo chuyen dong & Mo Gauss** | Train Augmentation | Mo phong xe chay toc do cao va thoi tiet suong mu. |
| **10** | **Cutout / Random Erasing** | Train Augmentation | Mo phong bien bao bi che khuat mot phan boi vat can. |

### B. Co che phuc hoi thich ung truoc suy luan (Adaptive Pre-Inference Restoration):
Khi nguoi dung tai len anh camera ngoai troi, he thong tu dong phan tich thong so anh va ap dung can thiep thich hop:
* **Phat hien nhieu hat ($\text{Residual} > 5.0$):** Chay Median Blur $3 \times 3$ de loai bo nhieu dom, giu nguyen khoang ho cua so 5, ngan chan viec bien 50 bi nhan dien nham thanh bien 80 hoac 120.
* **Phat hien thieu sang ($\mu < 95.0$):** Chay CLAHE tren kenh sang de khoi phuc nen trang va tach biet ro rang chu so den.
* **Phat hien mat net ($\text{Var}(\Delta I) < 50.0$):** Ap dung Unsharp Masking lam sac net bien ky tu.

---

## 4. HUONG DAN CHI TIET TU HUAN LUYEN LAI CAC MO HINH (RETRAINING GUIDE)

Moi kien truc trong he thong deu co the duoc huan luyen doc lap hoac huan luyen hang loat voi so luong Epoch tuy bien cao.

### 4.1. Chuan bi Moi truong

```bash
# Tao moi truong ao Python (Python 3.10 hoac 3.11)
python -m venv .venv

# Kich hoat moi truong ao tren Windows:
.venv\Scripts\activate

# Kich hoat moi truong ao tren Linux/macOS:
source .venv/bin/activate

# Cai dat toan bo thu vien phu thuoc:
pip install -r requirements.txt
```

---

### 4.2. Huong dan Huan luyen tung Mo hinh

#### [A] Huan luyen Custom CNN (Co Tang Cuong Du Lieu - Data Augmentation 2.0)
Day la mo hinh chu luc dat do chinh xac cao nhat va co kha nang khang nhieu manh me nhat.
```bash
# Huan luyen mac dinh (So epoch cao: 30 epochs, co EarlyStopping va ModelCheckpoint)
python -m src.train_cnn --retrain --epochs 30

# Huan luyen sau hon (50 epochs) voi co che can bang trong so Class Weighting:
python -m src.train_cnn --retrain --epochs 50

# Huan luyen kem tien xu ly Unsharp Masking va Gamma Correction:
python -m src.train_cnn --retrain --epochs 30 --use-gamma --use-unsharp
```
*Trọng số mo hinh duoc luu tai:* `models/custom_cnn_aug.keras` va `models/cnn_traffic_sign.keras`.

#### [B] Huan luyen Custom CNN Baseline (Khong Tang Cuong - Stage 1 Baseline)
Dung de lam doi chung danh gia hieu qua cua Data Augmentation.
```bash
# Huan luyen mo hinh Baseline khong co phep bien doi hinh hoc va ngoai canh:
python -m src.train_cnn --no-augment --retrain --epochs 30
```
*Trọng số mo hinh duoc luu tai:* `models/custom_cnn_no_aug.keras`.

#### [C] Huan luyen MobileNetV2 (Transfer Learning)
Mo hinh hoc sau sau chuyen giao tri thuc tu mang MobileNetV2 cua Google:
```bash
# Huan luyen MobileNetV2 voi 25 epochs:
python -m src.train_cnn --model-type mobilenet --retrain --epochs 25

# Huan luyen MobileNetV2 co tang cuong du lieu:
python -m src.train_cnn --model-type mobilenet --retrain --epochs 30
```
*Trọng số mo hinh duoc luu tai:* `models/mobilenet_no_aug.keras` hoac `models/mobilenet_aug.keras`.

#### [D] Huan luyen Machine Learning Co dien (HOG + SVM)
Mo hinh Machine Learning co dien trinh trich xuat 324 dac trung huong bien HOG va phan loai bang RBF SVM:
```bash
# Huan luyen HOG + SVM tren tap du lieu:
python -m src.train_svm
```
*Trọng số mo hinh duoc luu tai:* `models/svm_hog_model.pkl`.

#### [E] Huan luyen voi cac Ky thuat Lay mau Can bang (Imbalanced Sampling)
Neu muon thu nghiem cac ky thuat can bang du lieu thay the cho Class Weighting:
```bash
# Su dung thuat toan NearMiss-1:
python -m src.train_cnn --sampling nearmiss --retrain --epochs 25

# Su dung thuat toan Cluster Centroids (K-Means):
python -m src.train_cnn --sampling cluster_centroids --retrain --epochs 25

# Su dung Random Undersampling:
python -m src.train_cnn --sampling random_undersample --retrain --epochs 25
```

---

### 4.3. Danh gia Toan dien va Xuat Bang So sanh (Evaluation)

Sau khi huan luyen xong cac mo hinh, chay lenh danh gia doc lap tren 5.882 mau Test GTSRB:
```bash
python -m src.evaluate
```
Lenh nay se tu dong:
1. Danh gia tat ca 5 bien the mo hinh (Baseline CNN, MobileNetV2, HOG+SVM, Augmented CNN, Augmented SVM).
2. Tinh toan day du cac chi so: Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted).
3. Xuat file bang tong ket chuan hoc thuat tai `results/model_comparison.csv`.
4. Ve va xuat Ma tran nham lan (Confusion Matrix) tai `results/confusion_matrix_cnn.png` va `results/confusion_matrix_svm.png`.
5. Trich xuat cac cap bien bao nham lan tieu bieu va hinh anh du doan sai tai `results/misclassified/`.

---

### 4.4. Kiem thu Do ben vung truoc 6 Dieu kien Ngoai canh (Robustness Benchmark)

De do luong do suy giam hieu nang khi he thong hoat dong duoi troi mua, ban dem, suong mu, rung lac:
```bash
python -m src.robustness
```
Lenh tren se:
1. Mo phong lan luot 6 dieu kien ngoai canh tren 5 cap do nghiem trong (Severity 1 den 5).
2. So sanh truc tiep do ben giua mo hinh Chua Tang Cuong (No Aug) va Da Tang Cuong (Aug).
3. Do luong toc do suy luan thuc te (Latency tinh bang milliseconds va FPS) tren phan cung CPU.
4. Xuat bao cao tai `results/robustness_metrics.json` va `results/efficiency_metrics.json`.

---

## 5. KET QUA THUC NGHIEM VA DOI SANH HIEU NANG

Bang tong hop ket qua danh gia thuc te tren tap kiem thu doc lap GTSRB (5.882 mau) trich xuat truc tiep tu `results/model_comparison.csv`:

| Giai doan | Mo hinh | Tang cuong (Aug) | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Giai doan 1 (Baseline)** | Custom CNN (No Aug) | Khong | **99.22%** | 99.24% | 99.22% | 99.22% |
| **Giai doan 1 (Baseline)** | MobileNetV2 (No Aug) | Khong | **94.85%** | 95.10% | 94.85% | 94.92% |
| **Giai doan 1 (Baseline)** | HOG + SVM (No Aug) | Khong | **97.09%** | 97.13% | 97.09% | 97.09% |
| **Giai doan 2 (Tang cuong)** | **Custom CNN (Aug)** | **Co (6 dieu kien)** | **99.73%** | **99.73%** | **99.73%** | **99.73%** |
| **Giai doan 2 (Tang cuong)** | **HOG + SVM (Aug)** | **Co** | **97.82%** | 97.86% | 97.82% | 97.83% |

### Nhan xet Khoa hoc:
* **Custom CNN (Aug)** dat hieu nang vuot troi nhat toan dien voi **99.73% Accuracy** (chi nham 16 mau tren 5.882 anh test), khang nhieu tot truoc ca 6 dieu kien ngoai canh.
* **Kien truc 3 khoi Conv2D** toi uu hon han MobileNetV2 tren tap du lieu anh $32 \times 32$ nho tranh duoc hien tuong resolution bottleneck cua cac layer stride 2 lien tiep.
* **HOG + SVM** the hien suc manh xuat sac cua dac trung hinh hoc co dien, dat do chinh xac len toi **97.82%** ma khong can den mang no-ron phuc tap.

---

## 6. KHOI CHAY UNG DUNG WEB STREAMLIT

Sau khi huan luyen, khoi chay ung dung web tuong tac voi giao dien chuyen nghiep:

```bash
streamlit run app/app.py
```

### Kien truc 7 Tabs Chuyen Sau:
* **Tab 1: Tong quan & Pipeline (Dashboard):** Tong the kien truc, so do pipeline khong ro ri du lieu, thong so ky thuat va banner nhan dien thuong hieu.
* **Tab 2: Nhan dien & Camera Thoi gian thuc:** Ho tro 3 phuong thuc dau vao (Camera Webcam, Upload file anh va External Test). Ho tro 3 che do cat anh: Tu dong phat hien (Auto-detect), Cat thu cong (Manual Crop voi thanh truot % truc quan), va Giu nguyen anh goc.
* **Tab 3: Kham pha Tap du lieu (EDA):** Bieu do phan bo 43 lop, ty le mat can bang giua lop thieu so va da so, luoi anh mau truc quan hoa.
* **Tab 4: Tang cuong Du lieu & Thu thach Do ben:** Truc quan hoa 6 phep bien doi va phong thi nghiem truc tiep (Playground) cho phep nguoi dung chon bat ky bien bao nao (50, 80, 30, STOP...) va thu thach voi 6 loai nhieu o 5 cap do nghiem trong.
* **Tab 5: Danh gia & So sanh Mo hinh:** So sanh 3 nhom mo hinh theo tieu chuan nghien cuu (So sanh kien truc CNN, Doi chung SVM vs CNN, So sanh truoc va sau Tang cuong).
* **Tab 6: Ma tran Nham lan & Mo xe Loi sai:** Heatmap Confusion Matrix 43x43, thong ke Top 5 cap bien hay nham nhat va phong trung bay toan bo anh nham kem xac suat Softmax.
* **Tab 7: Nhat ky Du doan (SQLite Database):** Co so du lieu nhung `data/prediction_history.db` luu tru lich su thoi gian thuc, ho tro tim kiem va xuat bao cao dinh dang CSV.

---

## 7. CAU TRUC THU MUC DU AN

```text
traffic_sign/
├── app/
│   ├── app.py                      # Tap tin chinh dieu phoi giao dien Streamlit 7 Tabs
│   └── components/
│       └── ui_components.py        # Thu vien thanh phan giao dien (Header, Hero, Cards, Alerts)
├── data/
│   ├── raw/GTSRB/Train/            # Du lieu huan luyen GTSRB goc (39.209 anh, 43 thu muc lop)
│   ├── external_test/              # Anh chup thuc te ngoai camera de thu nghiem
│   └── prediction_history.db       # Co so du lieu SQLite luu nhat ky nhan dien
├── models/
│   ├── custom_cnn_aug.keras        # Custom CNN da qua tang cuong 6 dieu kien ngoai canh
│   ├── custom_cnn_no_aug.keras     # Custom CNN baseline chua tang cuong
│   ├── mobilenet_no_aug.keras      # MobileNetV2 Transfer Learning
│   ├── svm_hog_model.pkl           # Machine Learning HOG + SVM
│   └── model_metadata.json         # Thong so cau hinh tien xu ly & lich su huan luyen
├── notebooks/                      # 5 Jupyter Notebooks hoc thuat chuan muc
│   ├── 01_eda_and_data_exploration.ipynb
│   ├── 02_preprocessing_and_pipeline.ipynb
│   ├── 03_classical_ml_hog_svm.ipynb
│   ├── 04_deep_learning_custom_cnn.ipynb
│   └── 05_model_evaluation_and_comparison.ipynb
├── results/
│   ├── model_comparison.csv        # Bang so sanh chi so 3 giai doan
│   ├── robustness_metrics.json     # Do luong do ben vung duoi 6 loai nhieu
│   ├── efficiency_metrics.json     # Do tre suy luan (Latency & FPS)
│   ├── confusion_matrix_cnn.png    # Ma tran nham lan Custom CNN
│   ├── confusion_matrix_svm.png    # Ma tran nham lan HOG + SVM
│   └── training_accuracy.png       # Bieu do duong cong hoi tu do chinh xac
├── src/
│   ├── config.py                   # Thiet lap sieu tham so & duong dan he thong
│   ├── data_loader.py              # Bo nap du lieu tho & tao du lieu sample
│   ├── preprocessing.py            # 10 ky thuat tien xu ly & bo tang cuong du lieu 2.0
│   ├── feature_extraction.py       # Trinh trich xuat 324 chieu dac trung HOG
│   ├── train_cnn.py                # Pipeline huan luyen Custom CNN & MobileNetV2
│   ├── train_svm.py                # Pipeline huan luyen HOG + SVM
│   ├── evaluate.py                 # Danh gia doc lap 3 giai doan & phan tich loi sai
│   ├── robustness.py               # Mo phong 6 dieu kien ngoai canh & do FPS/Latency
│   ├── history_db.py               # Quan ly truy van CSDL SQLite nhat ky
│   └── predict.py                  # Pipeline suy luan phuc hoi thich ung thoi gian thuc
├── requirements.txt                # Danh muc thu vien Python
└── README.md                       # Tai lieu hoc thuat huong dan he thong
```

---

## THONG TIN LIEN HE & BAN QUYEN
* **Mon hoc:** May hoc (Machine Learning)
* **Khoa:** Cong nghe Thong tin
* **Nien khoa:** 2025 - 2026
