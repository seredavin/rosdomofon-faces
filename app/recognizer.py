"""Распознавание лиц моделью SFace (OpenCV FaceRecognizerSF) по базе образцов."""

import cv2
import numpy as np

from . import config
from .detector import Face, ensure_model
from .store import Gallery, Store


def match_gallery(gallery: Gallery, embedding: np.ndarray, threshold: float) -> tuple[int | None, float]:
    """Лучший человек по косинусному сходству (максимум по его образцам).

    Возвращает (person_id или None, если ниже порога / база пуста; сходство).
    """
    if gallery.matrix.shape[0] == 0:
        return None, 0.0
    similarities = gallery.matrix @ embedding
    best = int(np.argmax(similarities))
    similarity = float(similarities[best])
    if similarity < threshold:
        return None, similarity
    return int(gallery.person_ids[best]), similarity


class Recognizer:
    """Один экземпляр на воркер — FaceRecognizerSF не потокобезопасен.

    База образцов перечитывается из хранилища, когда меняется его версия, а порог
    читается на каждом сравнении — так изменения работают без перезапуска.
    """

    def __init__(self, store: Store) -> None:
        ensure_model(config.SFACE_MODEL_FILE, config.SFACE_MODEL_URL, config.SFACE_MODEL_SHA256)
        self._net = cv2.FaceRecognizerSF.create(str(config.SFACE_MODEL_FILE), "")
        self._store = store
        self._gallery: Gallery | None = None

    def align(self, frame: np.ndarray, face: Face) -> np.ndarray:
        """Выравнивает лицо по 5 опорным точкам в квадрат 112×112."""
        row = np.array(
            [face.x, face.y, face.w, face.h, *(c for p in face.landmarks for c in p), face.score],
            dtype=np.float32,
        )
        return self._net.alignCrop(frame, row)

    def embed(self, aligned: np.ndarray) -> np.ndarray:
        """Нормированный вектор признаков (128) выровненного лица."""
        feature = self._net.feature(aligned).reshape(-1).astype(np.float32)
        return feature / np.linalg.norm(feature)

    def match(self, embedding: np.ndarray) -> tuple[int | None, float]:
        return match_gallery(self._current(), embedding, self.threshold())

    def threshold(self) -> float:
        """Порог из настроек интерфейса, а если он не задан — из окружения."""
        threshold = self._store.match_threshold()
        return config.MATCH_THRESHOLD if threshold is None else threshold

    def name(self, person_id: int | None) -> str | None:
        if person_id is None:
            return None
        return self._current().names.get(person_id)

    def _current(self) -> Gallery:
        if self._gallery is None or self._gallery.version != self._store.version:
            self._gallery = self._store.gallery()
        return self._gallery
