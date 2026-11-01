"""Фильтр качества лица: годится ли кадр для распознавания и насколько он хорош."""

import math

import cv2
import numpy as np

from . import config
from .detector import Face

# Резкость, начиная с которой кадр считается «идеальным» для оценки качества
_SHARPNESS_FULL = 4
# Размер лица (px), начиная с которого он не добавляет к оценке — вход SFace 112×112
_SIZE_FULL = 112


def pose_metrics(face: Face) -> tuple[float, float]:
    """(смещение носа в долях межглазного расстояния, наклон линии глаз в градусах)."""
    eye_a, eye_b, nose = (np.array(p, dtype=np.float64) for p in face.landmarks[:3])
    eye_vec = eye_b - eye_a
    eye_dist = float(np.linalg.norm(eye_vec))
    if eye_dist < 1:
        return math.inf, 90.0
    offset = abs(float(np.dot(nose - (eye_a + eye_b) / 2, eye_vec / eye_dist))) / eye_dist
    tilt = math.degrees(math.atan2(abs(eye_vec[1]), abs(eye_vec[0])))
    return offset, tilt


def geometry_reason(face: Face) -> str | None:
    """Причина, по которой лицо не годится по геометрии (размер, поворот), или None."""
    if min(face.w, face.h) < config.MIN_FACE_SIZE:
        return "мелкое"
    offset, tilt = pose_metrics(face)
    if offset > config.MAX_NOSE_OFFSET:
        return "повёрнуто"
    if tilt > config.MAX_EYE_TILT_DEG:
        return "наклонено"
    return None


def sharpness(aligned: np.ndarray) -> float:
    """Резкость выровненного лица — дисперсия лапласиана."""
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_sharp(value: float) -> bool:
    return value >= config.MIN_SHARPNESS


def quality_score(face: Face, sharp: float) -> float:
    """Оценка 0..1 для выбора лучшего кадра появления."""
    offset, _ = pose_metrics(face)
    size = min(min(face.w, face.h) / _SIZE_FULL, 1.0)
    sharp_norm = min(sharp / (config.MIN_SHARPNESS * _SHARPNESS_FULL), 1.0)
    frontal = max(0.0, 1.0 - offset)
    return face.score * size * sharp_norm * frontal
