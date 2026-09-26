import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Path Configurations
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GTSRB_RAW_DIR = RAW_DATA_DIR / "GTSRB"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
EXTERNAL_TEST_DIR = DATA_DIR / "external_test"

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
MISCLASSIFIED_DIR = RESULT_DIR / "misclassified"
DB_PATH = DATA_DIR / "prediction_history.db"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, EXTERNAL_TEST_DIR, MODEL_DIR, RESULT_DIR, MISCLASSIFIED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Image & Model Parameters
IMAGE_SIZE = (32, 32)
IMAGE_CHANNELS = 3
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 43  # Default GTSRB, auto-adapts dynamically to any dataset size

# Dataset Split Percentages
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
RANDOM_STATE = 42

# Model & Evaluation Output Paths
SVM_MODEL_PATH = MODEL_DIR / "svm_hog_model.pkl"
CNN_MODEL_PATH = MODEL_DIR / "cnn_traffic_sign.keras"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"
TRAINING_HISTORY_PATH = MODEL_DIR / "training_history.json"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

COMPARISON_CSV_PATH = RESULT_DIR / "model_comparison.csv"
METRICS_JSON_PATH = RESULT_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = RESULT_DIR / "classification_report.json"
CONFUSION_MATRIX_CNN_PATH = RESULT_DIR / "confusion_matrix_cnn.png"
CONFUSION_MATRIX_SVM_PATH = RESULT_DIR / "confusion_matrix_svm.png"
TRAINING_ACCURACY_PATH = RESULT_DIR / "training_accuracy.png"
TRAINING_LOSS_PATH = RESULT_DIR / "training_loss.png"
DATASET_DISTRIBUTION_PATH = RESULT_DIR / "dataset_distribution.png"
PREPROCESSING_EXAMPLES_PATH = RESULT_DIR / "preprocessing_examples.png"

# GTSRB & Vietnamese Traffic Sign Class Name Dictionary Mapping
GTSRB_CLASSES = {
    0: {"vi": "Giới hạn tốc độ 20km/h (P.127)", "en": "Speed limit (20km/h)"},
    1: {"vi": "Giới hạn tốc độ 30km/h (P.127)", "en": "Speed limit (30km/h)"},
    2: {"vi": "Giới hạn tốc độ 50km/h (P.127)", "en": "Speed limit (50km/h)"},
    3: {"vi": "Giới hạn tốc độ 60km/h (P.127)", "en": "Speed limit (60km/h)"},
    4: {"vi": "Giới hạn tốc độ 70km/h (P.127)", "en": "Speed limit (70km/h)"},
    5: {"vi": "Giới hạn tốc độ 80km/h (P.127)", "en": "Speed limit (80km/h)"},
    6: {"vi": "Hết hạn chế tốc độ 80km/h", "en": "End of speed limit (80km/h)"},
    7: {"vi": "Giới hạn tốc độ 100km/h", "en": "Speed limit (100km/h)"},
    8: {"vi": "Giới hạn tốc độ 120km/h", "en": "Speed limit (120km/h)"},
    9: {"vi": "Cấm vượt (P.125)", "en": "No passing"},
    10: {"vi": "Cấm xe tải vượt (P.126)", "en": "No passing for vehicles over 3.5 metric tons"},
    11: {"vi": "Quyền ưu tiên tại giao lộ (W.207)", "en": "Right-of-way at next intersection"},
    12: {"vi": "Đường ưu tiên (I.401)", "en": "Priority road"},
    13: {"vi": "Nhường đường (W.208)", "en": "Yield"},
    14: {"vi": "Dừng lại (P.122 Stop)", "en": "Stop"},
    15: {"vi": "Cấm mọi phương tiện (P.101)", "en": "No vehicles"},
    16: {"vi": "Cấm xe tải (P.106a)", "en": "Vehicles over 3.5 metric tons prohibited"},
    17: {"vi": "Cấm đi ngược chiều (P.102)", "en": "No entry"},
    18: {"vi": "Cảnh báo nguy hiểm chung (W.233)", "en": "General caution"},
    19: {"vi": "Chỗ ngoặt nguy hiểm bên trái (W.201a)", "en": "Dangerous curve to the left"},
    20: {"vi": "Chỗ ngoặt nguy hiểm bên phải (W.201b)", "en": "Dangerous curve to the right"},
    21: {"vi": "Đoạn đường nhiều chỗ ngoặt (W.202)", "en": "Double curve"},
    22: {"vi": "Đường gồ gề (W.221a)", "en": "Bumpy road"},
    23: {"vi": "Đường trơn trượt (W.222a)", "en": "Slippery road"},
    24: {"vi": "Đường bị thu hẹp bên phải (W.203b)", "en": "Road narrows on the right"},
    25: {"vi": "Công trường (W.227)", "en": "Road work"},
    26: {"vi": "Tín hiệu đèn giao thông (W.209)", "en": "Traffic signals"},
    27: {"vi": "Người đi bộ cắt ngang (W.224)", "en": "Pedestrians"},
    28: {"vi": "Trẻ em sang đường (W.225)", "en": "Children crossing"},
    29: {"vi": "Xe đạp sang đường (W.226)", "en": "Bicycles crossing"},
    30: {"vi": "Cảnh báo băng tuyết (W.231)", "en": "Beware of ice/snow"},
    31: {"vi": "Động vật hoang dã cắt ngang (W.232)", "en": "Wild animals crossing"},
    32: {"vi": "Hết mọi lệnh cấm (DP.135)", "en": "End of all speed and passing limits"},
    33: {"vi": "Rẽ phải bắt buộc (R.301c)", "en": "Turn right ahead"},
    34: {"vi": "Rẽ trái bắt buộc (R.301d)", "en": "Turn left ahead"},
    35: {"vi": "Đi thẳng bắt buộc (R.301a)", "en": "Ahead only"},
    36: {"vi": "Đi thẳng hoặc rẽ phải (R.301d)", "en": "Go straight or right"},
    37: {"vi": "Đi thẳng hoặc rẽ trái (R.301e)", "en": "Go straight or left"},
    38: {"vi": "Đi bên phải (R.302a)", "en": "Keep right"},
    39: {"vi": "Đi bên trái (R.302b)", "en": "Keep left"},
    40: {"vi": "Vòng xuyến (R.303)", "en": "Roundabout mandatory"},
    41: {"vi": "Hết cấm vượt (DP.133)", "en": "End of no passing"},
    42: {"vi": "Hết cấm xe tải vượt (DP.134)", "en": "End of no passing by vehicles over 3.5 metric tons"}
}
