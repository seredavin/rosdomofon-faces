"""Пути, константы и настройки приложения.

Настройки читаются из переменных окружения с префиксом ROSDOMOFON_.
"""

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"ROSDOMOFON_{name}", default)


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_cameras(name: str) -> frozenset[int] | None:
    """Список id камер через запятую; пусто — все камеры аккаунта (None)."""
    raw = _env(name, "").strip()
    if not raw:
        return None
    return frozenset(int(part) for part in raw.split(",") if part.strip())


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_env("DATA_DIR", str(ROOT_DIR / "data")))
MODELS_DIR = ROOT_DIR / "models"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Файл с OAuth-токенами Росдомофона (создаётся с правами 0600)
TOKEN_FILE = DATA_DIR / "token.json"
# База людей, образцов и событий; снимки лиц — рядом в FACES_DIR
DB_FILE = DATA_DIR / "faces.db"
FACES_DIR = DATA_DIR / "faces"

# API Росдомофона (см. custom_components/rosdomofon/const.py в HA-интеграции)
BASE_URL = "https://rdba.rosdomofon.com"
SMS_REQUEST_URL = f"{BASE_URL}/abonents-service/api/v1/abonents/{{phone}}/sms"
TOKEN_REQUEST_URL = f"{BASE_URL}/authserver-service/oauth/token"
CAMERAS_LIST_URL = f"{BASE_URL}/abonents-service/api/v2/abonents/cameras"
KEYS_LIST_URL = f"{BASE_URL}/abonents-service/api/v2/abonents/keys"
OPEN_DOOR_URL = f"{BASE_URL}/rdas-service/api/v1/rdas/{{adapter_id}}/activate_key"
CLIENT_ID = "abonent"

# Модели из opencv_zoo: YuNet — детекция, SFace — распознавание
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
YUNET_MODEL_URL = f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_MODEL_FILE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_URL = f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx"
SFACE_MODEL_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
SFACE_MODEL_FILE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
# Шрифт с кириллицей для подписей на видео (OFL)
FONT_URL = (
    "https://github.com/notofonts/notofonts.github.io/raw/main/"
    "fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf"
)
FONT_SHA256 = "478c558ea716033cd60c03438f628dfa75694dcf6b5f6d505a2f05fd2b4f3823"
FONT_FILE = MODELS_DIR / "NotoSans-Regular.ttf"

# Детекция: ширина кадра, до которой он уменьшается перед поиском лиц,
# минимальная уверенность и максимальная частота запуска детектора.
DETECT_WIDTH = 640
DETECT_SCORE_THRESHOLD = 0.8
DETECT_MAX_FPS = 10

# Камеры круглосуточного мониторинга (None — все камеры аккаунта)
MONITOR_CAMERAS = _env_cameras("MONITOR_CAMERAS")

# Распознавание: порог косинусного сходства (0.363 — рекомендация авторов SFace)
MATCH_THRESHOLD = _env_float("MATCH_THRESHOLD", 0.363)
# Не чаще стольких распознаваний в секунду на одно появление
RECOGNIZE_MAX_FPS = 3
# Голосование: окно последних голосов и минимум голосов для вывода
VOTE_WINDOW = 7
VOTE_MIN = 3

# Фильтр качества
MIN_FACE_SIZE = _env_float("MIN_FACE_SIZE", 40)
MIN_SHARPNESS = _env_float("MIN_SHARPNESS", 30)
MAX_NOSE_OFFSET = 0.35  # смещение носа от середины глаз в долях межглазного расстояния
MAX_EYE_TILT_DEG = 25

# Трекинг: появление завершается, если лицо не видно дольше TRACK_TIMEOUT секунд
TRACK_TIMEOUT = _env_float("TRACK_TIMEOUT", 2)
TRACK_IOU_THRESHOLD = 0.3

# Сроки хранения событий (дни) и период очистки (секунды)
UNKNOWN_RETENTION_DAYS = _env_float("UNKNOWN_RETENTION_DAYS", 7)
KNOWN_RETENTION_DAYS = _env_float("KNOWN_RETENTION_DAYS", 30)
CLEANUP_INTERVAL = 3600

# Открытие двери: не чаще раза в столько секунд на дверь, таймаут запроса
UNLOCK_COOLDOWN = 30
UNLOCK_TIMEOUT = 5

# Воркер камеры вне мониторинга останавливается, если поток никто не смотрит столько секунд
WORKER_IDLE_TIMEOUT = 30
# Зритель считается активным столько секунд после последнего обращения
VIEWER_TIMEOUT = 5
JPEG_QUALITY = 80
