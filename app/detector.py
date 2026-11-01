"""Поиск лиц на кадре моделью YuNet (OpenCV FaceDetectorYN)."""

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

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


def ensure_model(path: Path, url: str, sha256: str) -> None:
    """Скачивает модель при первом запуске и проверяет контрольную сумму."""
    with _model_lock:
        if path.exists():
            return
        _LOGGER.info("Скачиваю модель: %s", url)
        resp = httpx.get(url, follow_redirects=True, timeout=120)
        resp.raise_for_status()
        digest = hashlib.sha256(resp.content).hexdigest()
        if digest != sha256:
            raise RuntimeError(f"Контрольная сумма модели {path.name} не совпала: {digest}")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(resp.content)


def ensure_models() -> None:
    """Скачивает все модели (детекция и распознавание)."""
    ensure_model(config.YUNET_MODEL_FILE, config.YUNET_MODEL_URL, config.YUNET_MODEL_SHA256)
    ensure_model(config.SFACE_MODEL_FILE, config.SFACE_MODEL_URL, config.SFACE_MODEL_SHA256)


class FaceDetector:
    """Обёртка над YuNet. Один экземпляр на воркер — FaceDetectorYN не потокобезопасен."""

    def __init__(self) -> None:
        ensure_model(config.YUNET_MODEL_FILE, config.YUNET_MODEL_URL, config.YUNET_MODEL_SHA256)
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
