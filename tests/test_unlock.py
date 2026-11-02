"""Тесты автоматического открытия двери."""

import pytest

from app.pipeline import FacePipeline
from app.rosdomofon import DoorError
from app.store import Store
from app.unlock import FAILED, OPENED, WOULD_OPEN, DoorUnlocker
from tests.test_pipeline import FRAME, FakeDetector, FakeRecognizer, _run
from tests.test_tracking import face

DOOR = ("aabbccddeeff", 1, "Дверь подъезда")


class FakeClient:
    def __init__(self, error: Exception | None = None):
        self.opened = []
        self.error = error

    def open_door(self, adapter_id, relay):
        if self.error:
            raise self.error
        self.opened.append((adapter_id, relay))


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "faces.db", tmp_path / "faces")
    yield s
    s.close()


@pytest.fixture
def person(store):
    person_id = store.create_person("Кирилл")
    store.set_can_unlock(person_id, True)
    store.set_camera_door(1001, DOOR)
    return person_id


def test_default_mode_is_dry_run(store, person):
    client = FakeClient()
    assert DoorUnlocker(client, store).try_unlock(1001, person) == WOULD_OPEN
    assert client.opened == []


def test_on_mode_opens_bound_door(store, person):
    store.set_unlock_mode("on")
    client = FakeClient()
    assert DoorUnlocker(client, store).try_unlock(1001, person) == OPENED
    assert client.opened == [("aabbccddeeff", 1)]


def test_off_mode_does_nothing(store, person):
    store.set_unlock_mode("off")
    client = FakeClient()
    assert DoorUnlocker(client, store).try_unlock(1001, person) is None
    assert client.opened == []


def test_unbound_camera_and_person_without_permission(store, person):
    store.set_unlock_mode("on")
    client = FakeClient()
    unlocker = DoorUnlocker(client, store)
    assert unlocker.try_unlock(1002, person) is None  # камера не привязана
    stranger = store.create_person("Сосед")
    assert unlocker.try_unlock(1001, stranger) is None  # нет разрешения
    assert client.opened == []


def test_cooldown_is_shared_between_cameras_of_one_door(store, person):
    store.set_unlock_mode("on")
    store.set_camera_door(1002, DOOR)
    client, clock = FakeClient(), Clock()
    unlocker = DoorUnlocker(client, store, clock=clock)
    assert unlocker.try_unlock(1001, person) == OPENED
    clock.now += 10
    assert unlocker.try_unlock(1002, person) is None
    clock.now += 25
    assert unlocker.try_unlock(1002, person) == OPENED
    assert len(client.opened) == 2


def test_dry_run_also_respects_cooldown(store, person):
    unlocker = DoorUnlocker(FakeClient(), store, clock=Clock())
    assert unlocker.try_unlock(1001, person) == WOULD_OPEN
    assert unlocker.try_unlock(1001, person) is None


def test_door_error_is_reported(store, person):
    store.set_unlock_mode("on")
    unlocker = DoorUnlocker(FakeClient(DoorError("HTTP 500")), store)
    assert unlocker.try_unlock(1001, person) == FAILED


# --- Хук в конвейере ---

def _pipeline(store, person_id, client):
    detector = FakeDetector()
    unlocker = DoorUnlocker(client, store)
    pipeline = FacePipeline(1001, store, detector, FakeRecognizer(store, person_id), unlocker)
    return pipeline, detector


def test_one_unlock_per_appearance_and_result_in_event(store, person):
    store.set_unlock_mode("on")
    client = FakeClient()
    pipeline, detector = _pipeline(store, person, client)
    detector.faces = [face(size=80)]
    _run(pipeline, FRAME, 60)  # минута у двери
    assert client.opened == [("aabbccddeeff", 1)]
    assert pipeline.faces()[0]["unlock"] == OPENED

    pipeline.flush()
    events, _ = store.list_events(unlocked=True)
    assert [(e["person_name"], e["unlock"]) for e in events] == [("Кирилл", OPENED)]


def test_no_unlock_before_voting_finishes(store, person):
    store.set_unlock_mode("on")
    client = FakeClient()
    pipeline, detector = _pipeline(store, person, client)
    detector.faces = [face(size=80)]
    # Распознавание не чаще 3 раз/с: за 0.5 с меньше 3 голосов — статус «определяется»
    _run(pipeline, FRAME, 0.5)
    assert pipeline.faces()[0]["status"] == "pending"
    assert client.opened == []


def test_unknown_person_never_unlocks(store, person):
    store.set_unlock_mode("on")
    client = FakeClient()
    pipeline, detector = _pipeline(store, None, client)
    detector.faces = [face(size=80)]
    _run(pipeline, FRAME, 5)
    assert client.opened == []


def test_unlocker_crash_does_not_break_pipeline(store, person):
    class Broken:
        def try_unlock(self, camera_id, person_id):
            raise RuntimeError("сбой")
    detector = FakeDetector()
    pipeline = FacePipeline(1001, store, detector, FakeRecognizer(store, person), Broken())
    detector.faces = [face(size=80)]
    _run(pipeline, FRAME, 2)
    assert pipeline.faces()[0]["unlock"] == FAILED
    pipeline.flush()
    assert len(store.list_events()[0]) == 1  # событие всё равно записано


def test_event_keeps_name_when_person_turns_away_after_unlock(store, person):
    """Сценарий из проверки 6.3: узнали, дверь открылась, человек отвернулся и ушёл."""
    store.set_unlock_mode("on")
    client = FakeClient()
    detector = FakeDetector()
    recognizer = FakeRecognizer(store, person)
    pipeline = FacePipeline(1001, store, detector, recognizer, DoorUnlocker(client, store))
    detector.faces = [face(size=80)]
    t = _run(pipeline, FRAME, 2)
    assert client.opened
    recognizer.person_id = None  # дальше лицо не совпадает с базой
    t = _run(pipeline, FRAME, 9, t0=t)
    assert pipeline.faces()[0]["status"] == "known"
    detector.faces = []
    _run(pipeline, FRAME, 3, t0=t)

    events, _ = store.list_events()
    assert [(e["person_name"], e["unlock"], e["assignable"]) for e in events] == [("Кирилл", "opened", False)]
