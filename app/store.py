"""Хранилище людей, образцов и событий: SQLite + JPEG-снимки на диске."""

import logging
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

_LOGGER = logging.getLogger(__name__)

# «Осиротевший» файл снимка удаляется, только если он старше этого возраста (сек):
# снимок пишется на диск до вставки строки в БД, свежий файл может быть ещё не связан.
_ORPHAN_MIN_AGE = 3600
_SECONDS_PER_DAY = 86400

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    id         INTEGER PRIMARY KEY,
    person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,
    image      TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY,
    camera_id  INTEGER NOT NULL,
    started_at REAL NOT NULL,
    ended_at   REAL NOT NULL,
    person_id  INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    image      TEXT NOT NULL,
    embedding  BLOB
);
CREATE INDEX IF NOT EXISTS events_started ON events(started_at);
CREATE INDEX IF NOT EXISTS events_person ON events(person_id);
CREATE INDEX IF NOT EXISTS samples_person ON samples(person_id);
"""


class StoreError(Exception):
    """Базовая ошибка хранилища."""


class NotFound(StoreError):
    pass


class Conflict(StoreError):
    pass


class Invalid(StoreError):
    pass


@dataclass(frozen=True)
class Gallery:
    """Снимок базы образцов для распознавателя: строки matrix соответствуют person_ids."""

    version: int
    person_ids: np.ndarray  # (N,) int64
    matrix: np.ndarray      # (N, 128) float32, строки нормированы
    names: dict[int, str]


def _to_blob(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).reshape(-1).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Store:
    """Потокобезопасное хранилище; одно соединение SQLite под блокировкой."""

    def __init__(self, db_file: Path = config.DB_FILE, faces_dir: Path = config.FACES_DIR) -> None:
        self._events_dir = faces_dir / "events"
        self._samples_dir = faces_dir / "samples"
        self._events_dir.mkdir(parents=True, exist_ok=True)
        self._samples_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_file, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)
        # Версия базы образцов: растёт при любом изменении людей/образцов
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- Люди ---

    def list_persons(self) -> list[dict]:
        with self._lock:
            persons = [dict(r) for r in self._db.execute(
                "SELECT id, name, created_at FROM persons ORDER BY name COLLATE NOCASE"
            )]
            samples: dict[int, list[dict]] = {}
            for r in self._db.execute("SELECT id, person_id, created_at FROM samples ORDER BY id"):
                samples.setdefault(r["person_id"], []).append({"id": r["id"], "created_at": r["created_at"]})
        for person in persons:
            person["samples"] = samples.get(person["id"], [])
        return persons

    def create_person(self, name: str) -> int:
        name = self._clean_name(name)
        with self._lock:
            try:
                cur = self._db.execute(
                    "INSERT INTO persons(name, created_at) VALUES (?, ?)", (name, time.time())
                )
            except sqlite3.IntegrityError as exc:
                raise Conflict(f"Человек с именем «{name}» уже есть") from exc
            self._version += 1
            return cur.lastrowid

    def rename_person(self, person_id: int, name: str) -> None:
        name = self._clean_name(name)
        with self._lock:
            try:
                cur = self._db.execute("UPDATE persons SET name = ? WHERE id = ?", (name, person_id))
            except sqlite3.IntegrityError as exc:
                raise Conflict(f"Человек с именем «{name}» уже есть") from exc
            if cur.rowcount == 0:
                raise NotFound("Человек не найден")
            self._version += 1

    def delete_person(self, person_id: int) -> None:
        with self._lock:
            images = [r["image"] for r in self._db.execute(
                "SELECT image FROM samples WHERE person_id = ?", (person_id,)
            )]
            cur = self._db.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            if cur.rowcount == 0:
                raise NotFound("Человек не найден")
            self._version += 1
        for image in images:
            (self._samples_dir / image).unlink(missing_ok=True)

    # --- Образцы ---

    def delete_sample(self, sample_id: int) -> None:
        with self._lock:
            row = self._db.execute("SELECT image FROM samples WHERE id = ?", (sample_id,)).fetchone()
            if row is None:
                raise NotFound("Образец не найден")
            self._db.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
            self._version += 1
        (self._samples_dir / row["image"]).unlink(missing_ok=True)

    def sample_image(self, sample_id: int) -> Path:
        with self._lock:
            row = self._db.execute("SELECT image FROM samples WHERE id = ?", (sample_id,)).fetchone()
        if row is None:
            raise NotFound("Образец не найден")
        return self._samples_dir / row["image"]

    def gallery(self) -> Gallery:
        """Все образцы одной матрицей — для сравнения с базой одним умножением."""
        with self._lock:
            version = self._version
            rows = self._db.execute("SELECT person_id, embedding FROM samples").fetchall()
            names = {r["id"]: r["name"] for r in self._db.execute("SELECT id, name FROM persons")}
        if rows:
            matrix = np.stack([_from_blob(r["embedding"]) for r in rows])
            matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        else:
            matrix = np.empty((0, 128), dtype=np.float32)
        person_ids = np.array([r["person_id"] for r in rows], dtype=np.int64)
        return Gallery(version=version, person_ids=person_ids, matrix=matrix.astype(np.float32), names=names)

    # --- События ---

    def add_event(
        self,
        camera_id: int,
        started_at: float,
        ended_at: float,
        person_id: int | None,
        image_jpeg: bytes,
        embedding: np.ndarray | None,
    ) -> int:
        """Записывает событие. Вектор хранится только у событий незнакомых."""
        image = f"{uuid.uuid4().hex}.jpg"
        (self._events_dir / image).write_bytes(image_jpeg)
        blob = _to_blob(embedding) if person_id is None and embedding is not None else None
        with self._lock:
            try:
                cur = self._db.execute(
                    "INSERT INTO events(camera_id, started_at, ended_at, person_id, image, embedding)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (camera_id, started_at, ended_at, person_id, image, blob),
                )
            except sqlite3.IntegrityError:
                # Человека удалили, пока шло появление — пишем как незнакомого без вектора
                cur = self._db.execute(
                    "INSERT INTO events(camera_id, started_at, ended_at, person_id, image, embedding)"
                    " VALUES (?, ?, ?, NULL, ?, NULL)",
                    (camera_id, started_at, ended_at, image),
                )
            return cur.lastrowid

    def list_events(
        self,
        *,
        camera_id: int | None = None,
        person_id: int | None = None,
        unknown: bool = False,
        gallery: bool = False,
        since: float | None = None,
        until: float | None = None,
        before_id: int | None = None,
        limit: int = 50,
    ) -> tuple[list[dict], int | None]:
        """События от новых к старым; возвращает (страница, before_id следующей страницы или None).

        gallery=True — только карточки незнакомых, которые можно подписать (есть вектор).
        """
        where, args = [], []
        if camera_id is not None:
            where.append("e.camera_id = ?")
            args.append(camera_id)
        if person_id is not None:
            where.append("e.person_id = ?")
            args.append(person_id)
        if unknown or gallery:
            where.append("e.person_id IS NULL")
        if gallery:
            where.append("e.embedding IS NOT NULL")
        if since is not None:
            where.append("e.started_at >= ?")
            args.append(since)
        if until is not None:
            where.append("e.started_at < ?")
            args.append(until)
        if before_id is not None:
            where.append("e.id < ?")
            args.append(before_id)
        sql = (
            "SELECT e.id, e.camera_id, e.started_at, e.ended_at, e.person_id, p.name AS person_name,"
            " e.embedding IS NOT NULL AS assignable"
            " FROM events e LEFT JOIN persons p ON p.id = e.person_id"
            + (" WHERE " + " AND ".join(where) if where else "")
            + " ORDER BY e.id DESC LIMIT ?"
        )
        with self._lock:
            rows = [dict(r) for r in self._db.execute(sql, (*args, limit + 1))]
        for row in rows:
            row["assignable"] = bool(row["assignable"])
        has_more = len(rows) > limit
        rows = rows[:limit]
        return rows, (rows[-1]["id"] if has_more else None)

    def event_image(self, event_id: int) -> Path:
        with self._lock:
            row = self._db.execute("SELECT image FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            raise NotFound("Событие не найдено")
        return self._events_dir / row["image"]

    def delete_event(self, event_id: int) -> None:
        with self._lock:
            row = self._db.execute("SELECT image FROM events WHERE id = ?", (event_id,)).fetchone()
            if row is None:
                raise NotFound("Событие не найдено")
            self._db.execute("DELETE FROM events WHERE id = ?", (event_id,))
        (self._events_dir / row["image"]).unlink(missing_ok=True)

    def assign_event(self, event_id: int, *, person_id: int | None = None, name: str | None = None) -> int:
        """Подписывает карточку незнакомого: лицо становится образцом человека.

        Указывается либо person_id существующего человека, либо name нового.
        Возвращает id человека.
        """
        if (person_id is None) == (name is None):
            raise Invalid("Нужно указать либо существующего человека, либо имя нового")
        with self._lock:
            event = self._db.execute(
                "SELECT image, embedding, person_id FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            if event is None:
                raise NotFound("Событие не найдено")
            if event["person_id"] is not None or event["embedding"] is None:
                raise Conflict("Эту карточку нельзя подписать: имя уже назначено или нет данных лица")
            sample_image = f"{uuid.uuid4().hex}.jpg"
            shutil.copyfile(self._events_dir / event["image"], self._samples_dir / sample_image)
            try:
                self._db.execute("BEGIN")
                if name is not None:
                    name = self._clean_name(name)
                    try:
                        person_id = self._db.execute(
                            "INSERT INTO persons(name, created_at) VALUES (?, ?)", (name, time.time())
                        ).lastrowid
                    except sqlite3.IntegrityError as exc:
                        raise Conflict(f"Человек с именем «{name}» уже есть") from exc
                elif self._db.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,)).fetchone() is None:
                    raise NotFound("Человек не найден")
                self._db.execute(
                    "INSERT INTO samples(person_id, embedding, image, created_at) VALUES (?, ?, ?, ?)",
                    (person_id, event["embedding"], sample_image, time.time()),
                )
                self._db.execute(
                    "UPDATE events SET person_id = ?, embedding = NULL WHERE id = ?", (person_id, event_id)
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                (self._samples_dir / sample_image).unlink(missing_ok=True)
                raise
            self._version += 1
            return person_id

    # --- Очистка ---

    def cleanup(
        self,
        now: float | None = None,
        unknown_days: float = config.UNKNOWN_RETENTION_DAYS,
        known_days: float = config.KNOWN_RETENTION_DAYS,
    ) -> int:
        """Удаляет события старше сроков хранения и «осиротевшие» снимки. Возвращает число событий."""
        now = time.time() if now is None else now
        with self._lock:
            expired = self._db.execute(
                "SELECT id, image FROM events WHERE (person_id IS NULL AND ended_at < ?)"
                " OR (person_id IS NOT NULL AND ended_at < ?)",
                (now - unknown_days * _SECONDS_PER_DAY, now - known_days * _SECONDS_PER_DAY),
            ).fetchall()
            self._db.executemany("DELETE FROM events WHERE id = ?", [(r["id"],) for r in expired])
            used_events = {r["image"] for r in self._db.execute("SELECT image FROM events")}
            used_samples = {r["image"] for r in self._db.execute("SELECT image FROM samples")}
        for row in expired:
            (self._events_dir / row["image"]).unlink(missing_ok=True)
        self._remove_orphans(self._events_dir, used_events, now)
        self._remove_orphans(self._samples_dir, used_samples, now)
        if expired:
            _LOGGER.info("Очистка: удалено событий — %d", len(expired))
        return len(expired)

    # --- Внутреннее ---

    @staticmethod
    def _remove_orphans(directory: Path, used: set[str], now: float) -> None:
        for path in directory.glob("*.jpg"):
            if path.name not in used and now - path.stat().st_mtime > _ORPHAN_MIN_AGE:
                path.unlink(missing_ok=True)

    @staticmethod
    def _clean_name(name: str) -> str:
        name = (name or "").strip()
        if not name:
            raise Invalid("Имя не может быть пустым")
        if len(name) > 100:
            raise Invalid("Имя длиннее 100 символов")
        return name
