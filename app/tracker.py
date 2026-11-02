"""Трекинг лиц между кадрами и голосование за имя появления."""

import itertools
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from . import config
from .detector import Face

PENDING = "pending"
KNOWN = "known"
UNKNOWN = "unknown"


def iou(a: Face, b: Face) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """Одно появление лица в кадре."""

    id: int
    face: Face
    started_at: float           # время начала (UTC epoch)
    last_seen: float            # последнее обнаружение (monotonic)
    ended_at: float             # последнее обнаружение (UTC epoch)
    status: str = PENDING
    person_id: int | None = None
    votes: deque = field(default_factory=lambda: deque(maxlen=config.VOTE_WINDOW))
    last_recognized: float = -1e9
    best_score: float = -1.0
    best_image: bytes | None = None
    best_embedding: np.ndarray | None = None
    # Результат автоматического открытия двери; проверка — один раз за появление
    unlock_result: str | None = None
    unlock_checked: bool = False

    @property
    def had_usable_face(self) -> bool:
        return self.best_image is not None

    def needs_recognition(self, now: float) -> bool:
        """Не чаще RECOGNIZE_MAX_FPS распознаваний в секунду на появление."""
        return now - self.last_recognized >= 1 / config.RECOGNIZE_MAX_FPS

    def add_vote(self, person_id: int | None, now: float) -> None:
        """Голос «человек person_id» или «незнакомый» (None); пересчитывает статус.

        Статус — большинство среди последних VOTE_WINDOW голосов, но не раньше
        VOTE_MIN голосов. При равенстве остаётся прежнее решение.
        """
        self.last_recognized = now
        self.votes.append(person_id)
        if len(self.votes) < config.VOTE_MIN:
            return
        ranked = Counter(self.votes).most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return
        winner = ranked[0][0]
        self.person_id = winner
        self.status = UNKNOWN if winner is None else KNOWN

    def final_person(self) -> int | None:
        """Итог появления для журнала: решение голосования, а если голосов
        не хватило — однозначное большинство имеющихся, иначе «незнакомый»."""
        if self.status != PENDING:
            return self.person_id
        ranked = Counter(self.votes).most_common()
        if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
            return None
        return ranked[0][0]

    def offer_best(self, score: float, image: bytes, embedding: np.ndarray) -> None:
        if score > self.best_score:
            self.best_score = score
            self.best_image = image
            self.best_embedding = embedding


class Tracker:
    """Жадное сопоставление детекций с активными появлениями по IoU рамок."""

    def __init__(self) -> None:
        self._tracks: list[Track] = []
        self._ids = itertools.count(1)

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def update(self, faces: list[Face], now: float, wall: float) -> tuple[list[Track], list[Track]]:
        """Обновляет треки детекциями кадра.

        Возвращает (трек для каждого лица в порядке faces, завершённые треки).
        """
        pairs = sorted(
            ((iou(t.face, f), ti, fi) for ti, t in enumerate(self._tracks) for fi, f in enumerate(faces)),
            reverse=True,
        )
        assigned: dict[int, Track] = {}
        used_tracks: set[int] = set()
        for score, ti, fi in pairs:
            if score < config.TRACK_IOU_THRESHOLD:
                break
            if ti in used_tracks or fi in assigned:
                continue
            used_tracks.add(ti)
            assigned[fi] = self._tracks[ti]

        matched = []
        for fi, face in enumerate(faces):
            track = assigned.get(fi)
            if track is None:
                track = Track(id=next(self._ids), face=face, started_at=wall, last_seen=now, ended_at=wall)
                self._tracks.append(track)
            track.face = face
            track.last_seen = now
            track.ended_at = wall
            matched.append(track)

        finished = [t for t in self._tracks if now - t.last_seen > config.TRACK_TIMEOUT]
        if finished:
            self._tracks = [t for t in self._tracks if t not in finished]
        return matched, finished

    def flush(self) -> list[Track]:
        """Завершает все активные треки (поток камеры закрывается)."""
        finished, self._tracks = self._tracks, []
        return finished
