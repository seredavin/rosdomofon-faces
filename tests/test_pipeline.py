"""Тесты конвейера кадра на подставных детекторе и распознавателе."""

import numpy as np
import pytest

from app.pipeline import FacePipeline
from app.store import Store
from tests.test_tracking import face


class FakeDetector:
    def __init__(self):
        self.faces = []

    def detect(self, frame):
        return list(self.faces)


class FakeRecognizer:
    """Узнаёт всех как person_id (None — незнакомый); кадр всегда резкий."""

    def __init__(self, store, person_id=None):
        self.store = store
        self.person_id = person_id

    def align(self, frame, f):
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)

    def embed(self, aligned):
        v = np.ones(128, np.float32)
        return v / np.linalg.norm(v)

    def match(self, embedding):
        return self.person_id, 0.9

    def name(self, person_id):
        return {p["id"]: p["name"] for p in self.store.list_persons()}.get(person_id)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "faces.db", tmp_path / "faces")
    yield s
    s.close()


def _run(pipeline, frame, seconds, t0=0.0, step=0.05):
    t = t0
    while t < t0 + seconds:
        pipeline.process(frame, t, 1_000_000 + t)
        t += step
    return t


FRAME = np.zeros((480, 704, 3), np.uint8)


def test_unknown_appearance_creates_one_gallery_event(store):
    detector = FakeDetector()
    pipeline = FacePipeline(1001, store, detector, FakeRecognizer(store))
    detector.faces = [face(size=80)]
    t = _run(pipeline, FRAME, 3)
    assert pipeline.faces()[0]["status"] == "unknown"
    detector.faces = []
    _run(pipeline, FRAME, 3, t0=t)

    events, _ = store.list_events()
    assert len(events) == 1
    assert events[0]["person_id"] is None and events[0]["assignable"]
    assert events[0]["ended_at"] - events[0]["started_at"] == pytest.approx(3, abs=0.2)


def test_known_person_event_and_api_shape(store):
    person_id = store.create_person("Кирилл")
    detector = FakeDetector()
    pipeline = FacePipeline(1001, store, detector, FakeRecognizer(store, person_id))
    detector.faces = [face(size=80)]
    _run(pipeline, FRAME, 2)

    [current] = pipeline.faces()
    assert current["status"] == "known"
    assert current["person"] == {"id": person_id, "name": "Кирилл"}
    assert {"track_id", "x", "y", "w", "h"} <= current.keys()

    pipeline.flush()
    events, _ = store.list_events()
    assert events[0]["person_name"] == "Кирилл"
    assert not events[0]["assignable"]  # у знакомых вектор не хранится


def test_small_faces_only_produce_no_event(store):
    detector = FakeDetector()
    pipeline = FacePipeline(1001, store, detector, FakeRecognizer(store))
    detector.faces = [face(size=20)]
    _run(pipeline, FRAME, 2)
    assert pipeline.faces()[0]["status"] == "pending"
    pipeline.flush()
    assert store.list_events()[0] == []


def test_draw_does_not_fail_near_frame_edges(store):
    detector = FakeDetector()
    pipeline = FacePipeline(1001, store, detector, FakeRecognizer(store))
    detector.faces = [face(x=0, y=0, size=80), face(x=650, y=420, size=80)]
    _run(pipeline, FRAME, 1)
    frame = FRAME.copy()
    pipeline.draw(frame)
    assert frame.any()
