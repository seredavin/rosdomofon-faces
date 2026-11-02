"""Конвейер кадра: детекция → трекинг → качество → распознавание → голосование → события."""

import logging
import threading

import cv2
import numpy as np

from . import config
from .detector import Face, FaceDetector
from .overlay import draw_box
from .quality import geometry_reason, is_sharp, quality_score, sharpness
from .recognizer import Recognizer
from .store import Store
from .tracker import KNOWN, PENDING, Track, Tracker

_LOGGER = logging.getLogger(__name__)

# Снимок лица для журнала: рамка, расширенная в столько раз, и качество JPEG
_SNAPSHOT_MARGIN = 1.6
_SNAPSHOT_QUALITY = 90

_LABELS = {PENDING: "…", "unknown": "Неизвестный"}


def crop_snapshot(frame: np.ndarray, face: Face) -> bytes:
    """Вырезка лица с полями для снимка в журнале (JPEG)."""
    cx, cy = face.x + face.w / 2, face.y + face.h / 2
    half = max(face.w, face.h) * _SNAPSHOT_MARGIN / 2
    fh, fw = frame.shape[:2]
    x1, y1 = max(int(cx - half), 0), max(int(cy - half), 0)
    x2, y2 = min(int(cx + half), fw), min(int(cy + half), fh)
    ok, buf = cv2.imencode(".jpg", frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, _SNAPSHOT_QUALITY])
    if not ok:
        raise RuntimeError("Не удалось закодировать снимок лица")
    return buf.tobytes()


class FacePipeline:
    """Анализ кадров одной камеры. Вызывается из потока воркера;
    faces() безопасно читать из других потоков."""

    def __init__(
        self,
        camera_id: int,
        store: Store,
        detector: FaceDetector | None = None,
        recognizer: Recognizer | None = None,
    ) -> None:
        self._camera_id = camera_id
        self._store = store
        self._detector = detector or FaceDetector()
        self._recognizer = recognizer or Recognizer(store)
        self._tracker = Tracker()
        self._last_detect = -1e9
        self._visible: list[Track] = []
        self._snapshot: list[dict] = []
        self._lock = threading.Lock()

    def process(self, frame: np.ndarray, now: float, wall: float) -> None:
        """Обрабатывает кадр; детекция — не чаще DETECT_MAX_FPS."""
        if now - self._last_detect < 1 / config.DETECT_MAX_FPS:
            return
        self._last_detect = now
        faces = self._detector.detect(frame)
        visible, finished = self._tracker.update(faces, now, wall)
        for track in visible:
            self._analyze(frame, track, now)
        for track in finished:
            self._finish(track)
        self._visible = visible
        snapshot = [self._describe(t) for t in visible]
        with self._lock:
            self._snapshot = snapshot

    def draw(self, frame: np.ndarray) -> None:
        """Рамки и подписи текущих лиц поверх кадра."""
        for track in self._visible:
            face = track.face
            draw_box(frame, (face.x, face.y, face.w, face.h), track.status, self._label(track))

    def faces(self) -> list[dict]:
        with self._lock:
            return list(self._snapshot)

    def flush(self) -> None:
        """Завершает все появления (поток закрывается) и записывает события."""
        for track in self._tracker.flush():
            self._finish(track)
        self._visible = []
        with self._lock:
            self._snapshot = []

    # --- Внутреннее ---

    def _analyze(self, frame: np.ndarray, track: Track, now: float) -> None:
        if not track.needs_recognition(now):
            return
        track.last_recognized = now  # частота ограничивается и для неудачных попыток
        if geometry_reason(track.face) is not None:
            return
        aligned = self._recognizer.align(frame, track.face)
        sharp = sharpness(aligned)
        if not is_sharp(sharp):
            return
        embedding = self._recognizer.embed(aligned)
        person_id, _similarity = self._recognizer.match(embedding)
        track.add_vote(person_id, now)
        score = quality_score(track.face, sharp)
        if score > track.best_score:
            track.offer_best(score, crop_snapshot(frame, track.face), embedding)

    def _finish(self, track: Track) -> None:
        if not track.had_usable_face:
            return
        person_id = track.final_person()
        try:
            event_id = self._store.add_event(
                self._camera_id, track.started_at, track.ended_at,
                person_id, track.best_image, track.best_embedding,
            )
        except Exception:  # noqa: BLE001 — сбой записи не должен ронять анализ камеры
            _LOGGER.exception("Камера %s: не удалось записать событие", self._camera_id)
            return
        _LOGGER.info(
            "Камера %s: событие %d — %s", self._camera_id, event_id,
            self._recognizer.name(person_id) or "Неизвестный",
        )

    def _label(self, track: Track) -> str:
        if track.status == KNOWN:
            return self._recognizer.name(track.person_id) or "?"
        return _LABELS[track.status]

    def _describe(self, track: Track) -> dict:
        face = track.face
        person = None
        if track.status == KNOWN:
            person = {"id": track.person_id, "name": self._recognizer.name(track.person_id)}
        return {
            "track_id": track.id,
            "x": face.x, "y": face.y, "w": face.w, "h": face.h,
            "score": face.score,
            "status": track.status,
            "person": person,
        }
