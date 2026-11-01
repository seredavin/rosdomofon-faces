"""Поиск лиц на кадре моделью YuNet (OpenCV FaceDetectorYN)."""

import hashlib
import logging
import threading
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

from . import config

_LOGGER = logging.getLogger(__name__)
_model_lock = threading.Lock()


@dataclass(frozen=True)
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float
    # 5 опорных точек (глаза, нос, уголки рта) — пригодятся для распознавания
    landmarks: tuple[tuple[int, int], ...]


def ensure_model() -> None:
    """Скачивает модель YuNet при первом запуске и проверяет контрольную сумму."""
    path = config.YUNET_MODEL_FILE
    with _model_lock:
        if path.exists():
            return
        _download_model(path)


def _download_model(path) -> None:
    _LOGGER.info("Скачиваю модель детекции лиц: %s", config.YUNET_MODEL_URL)
    resp = httpx.get(config.YUNET_MODEL_URL, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    digest = hashlib.sha256(resp.content).hexdigest()
    if digest != config.YUNET_MODEL_SHA256:
        raise RuntimeError(f"Контрольная сумма модели не совпала: {digest}")
    config.MODELS_DIR.mkdir(exist_ok=True)
    path.write_bytes(resp.content)


class FaceDetector:
    """Обёртка над YuNet. Один экземпляр на воркер — FaceDetectorYN не потокобезопасен."""

    def __init__(self) -> None:
        ensure_model()
        self._net = cv2.FaceDetectorYN.create(
            str(config.YUNET_MODEL_FILE), "", (320, 320), config.DETECT_SCORE_THRESHOLD
        )
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[Face]:
        """Находит лица; кадр уменьшается до DETECT_WIDTH, координаты возвращаются в исходном масштабе."""
        h, w = frame.shape[:2]
        scale = min(1.0, config.DETECT_WIDTH / w)
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
        size = (small.shape[1], small.shape[0])
        if size != self._input_size:
            self._net.setInputSize(size)
            self._input_size = size

        _, raw = self._net.detect(small)
        if raw is None:
            return []

        faces = []
        for row in raw:
            box = row[:4] / scale
            points = row[4:14].reshape(5, 2) / scale
            faces.append(Face(
                x=int(box[0]), y=int(box[1]), w=int(box[2]), h=int(box[3]),
                score=float(row[14]),
                landmarks=tuple((int(px), int(py)) for px, py in points),
            ))
        return faces


def draw_faces(frame: np.ndarray, faces: list[Face]) -> None:
    """Рисует рамки и уверенность поверх кадра (in-place)."""
    for face in faces:
        cv2.rectangle(frame, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 220, 0), 2)
        cv2.putText(
            frame, f"{face.score:.2f}", (face.x, max(face.y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2,
        )
