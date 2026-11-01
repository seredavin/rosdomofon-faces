"""Пути и константы приложения."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Файл с OAuth-токенами Росдомофона (создаётся с правами 0600)
TOKEN_FILE = DATA_DIR / "token.json"

# API Росдомофона (см. custom_components/rosdomofon/const.py в HA-интеграции)
BASE_URL = "https://rdba.rosdomofon.com"
SMS_REQUEST_URL = f"{BASE_URL}/abonents-service/api/v1/abonents/{{phone}}/sms"
TOKEN_REQUEST_URL = f"{BASE_URL}/authserver-service/oauth/token"
CAMERAS_LIST_URL = f"{BASE_URL}/abonents-service/api/v2/abonents/cameras"
CLIENT_ID = "abonent"

# Модель детекции лиц YuNet из opencv_zoo
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_MODEL_FILE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

# Детекция: ширина кадра, до которой он уменьшается перед поиском лиц,
# минимальная уверенность и максимальная частота запуска детектора.
DETECT_WIDTH = 640
DETECT_SCORE_THRESHOLD = 0.8
DETECT_MAX_FPS = 10

# Воркер камеры останавливается, если поток никто не смотрит столько секунд
WORKER_IDLE_TIMEOUT = 30
JPEG_QUALITY = 80
