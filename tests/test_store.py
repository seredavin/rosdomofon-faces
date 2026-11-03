"""Тесты хранилища людей, образцов и событий."""

import os
import time

import numpy as np
import pytest

from app.store import Conflict, Invalid, NotFound, Store

DAY = 86400


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "faces.db", tmp_path / "faces")
    yield s
    s.close()


def _vec(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=128).astype(np.float32)


def _unknown_event(store: Store, seed: int = 1, ended_at: float | None = None) -> int:
    t = time.time() if ended_at is None else ended_at
    return store.add_event(1001, t - 5, t, None, b"jpeg", _vec(seed))


def test_create_person_rejects_empty_and_duplicate(store):
    store.create_person("Кирилл")
    with pytest.raises(Invalid):
        store.create_person("   ")
    with pytest.raises(Conflict):
        store.create_person(" Кирилл ")


def test_rename_and_delete_missing_person(store):
    with pytest.raises(NotFound):
        store.rename_person(999, "Имя")
    with pytest.raises(NotFound):
        store.delete_person(999)


def test_assign_to_new_person_creates_sample(store):
    event_id = _unknown_event(store)
    version = store.version

    person_id = store.assign_event(event_id, name="Кирилл")

    persons = store.list_persons()
    assert [p["name"] for p in persons] == ["Кирилл"]
    assert len(persons[0]["samples"]) == 1
    assert store.version > version
    events, _ = store.list_events()
    assert events[0]["person_id"] == person_id
    assert events[0]["person_name"] == "Кирилл"
    assert not events[0]["assignable"]
    # Снимок образца — отдельная копия
    sample_id = persons[0]["samples"][0]["id"]
    assert store.sample_image(sample_id).read_bytes() == b"jpeg"
    assert store.sample_image(sample_id) != store.event_image(event_id)


def test_assign_to_existing_person_adds_sample(store):
    person_id = store.assign_event(_unknown_event(store, 1), name="Кирилл")
    store.assign_event(_unknown_event(store, 2), person_id=person_id)
    assert len(store.list_persons()[0]["samples"]) == 2


def test_assign_twice_or_with_taken_name_fails(store):
    store.create_person("Кирилл")
    event_id = _unknown_event(store)
    with pytest.raises(Conflict):
        store.assign_event(event_id, name="Кирилл")
    # Неудачное назначение ничего не меняет: карточку можно подписать позже
    assert store.list_events(gallery=True)[0][0]["id"] == event_id
    store.assign_event(event_id, name="Аня")
    with pytest.raises(Conflict):
        store.assign_event(event_id, name="Петя")


def test_known_event_has_no_embedding(store):
    person_id = store.create_person("Кирилл")
    store.add_event(1001, 0, 1, person_id, b"jpeg", _vec(1))
    assert store.list_events(gallery=True)[0] == []


def test_delete_person_cascades_samples_and_unlinks_events(store):
    event_id = _unknown_event(store)
    person_id = store.assign_event(event_id, name="Кирилл")
    sample_path = store.sample_image(store.list_persons()[0]["samples"][0]["id"])

    store.delete_person(person_id)

    assert store.list_persons() == []
    assert not sample_path.exists()
    assert store.gallery().matrix.shape == (0, 128)
    events, _ = store.list_events()
    assert events[0]["person_id"] is None
    # Событие без вектора в галерею не возвращается
    assert store.list_events(gallery=True)[0] == []


def test_delete_sample(store):
    store.assign_event(_unknown_event(store), name="Кирилл")
    sample_id = store.list_persons()[0]["samples"][0]["id"]
    store.delete_sample(sample_id)
    assert store.list_persons()[0]["samples"] == []
    with pytest.raises(NotFound):
        store.delete_sample(sample_id)


def test_gallery_matrix_is_normalized(store):
    store.assign_event(_unknown_event(store, 1), name="Кирилл")
    store.assign_event(_unknown_event(store, 2), name="Аня")
    gallery = store.gallery()
    assert gallery.matrix.shape == (2, 128)
    np.testing.assert_allclose(np.linalg.norm(gallery.matrix, axis=1), 1, rtol=1e-5)
    assert set(gallery.names.values()) == {"Кирилл", "Аня"}


def test_list_events_filters_and_pagination(store):
    person_id = store.create_person("Кирилл")
    ids = [store.add_event(1001 if i % 2 else 1002, i, i + 1, person_id if i < 3 else None, b"j", _vec(i))
           for i in range(7)]

    page, next_id = store.list_events(limit=3)
    assert [e["id"] for e in page] == ids[::-1][:3]
    page2, _ = store.list_events(limit=3, before_id=next_id)
    assert [e["id"] for e in page2] == ids[::-1][3:6]

    assert {e["camera_id"] for e in store.list_events(camera_id=1002)[0]} == {1002}
    assert len(store.list_events(person_id=person_id)[0]) == 3
    assert len(store.list_events(unknown=True)[0]) == 4
    assert [e["id"] for e in store.list_events(since=2, until=4)[0]] == [ids[3], ids[2]]
    assert store.list_events(limit=50)[1] is None


def test_delete_event_removes_image(store):
    event_id = _unknown_event(store)
    path = store.event_image(event_id)
    store.delete_event(event_id)
    assert not path.exists()
    with pytest.raises(NotFound):
        store.event_image(event_id)


def test_cleanup_respects_retention_and_keeps_samples(store):
    now = time.time()
    old_unknown = _unknown_event(store, 1, ended_at=now - 8 * DAY)
    fresh_unknown = _unknown_event(store, 2, ended_at=now - 1 * DAY)
    # Карточку подписали — событие стало «знакомым», срок 30 дней
    assigned = _unknown_event(store, 3, ended_at=now - 10 * DAY)
    store.assign_event(assigned, name="Кирилл")
    very_old_known = _unknown_event(store, 4, ended_at=now - 31 * DAY)
    store.assign_event(very_old_known, name="Аня")

    removed = store.cleanup(now=now, unknown_days=7, known_days=30)

    assert removed == 2
    remaining = {e["id"] for e in store.list_events()[0]}
    assert remaining == {fresh_unknown, assigned}
    assert old_unknown not in remaining
    # Образцы обоих людей остались, включая Аню, чьё событие удалено
    assert all(len(p["samples"]) == 1 for p in store.list_persons())
    assert store.gallery().matrix.shape == (2, 128)


def test_cleanup_removes_old_orphan_files_only(store, tmp_path):
    events_dir = tmp_path / "faces" / "events"
    old_orphan = events_dir / "old.jpg"
    new_orphan = events_dir / "new.jpg"
    old_orphan.write_bytes(b"x")
    new_orphan.write_bytes(b"x")
    linked = store.event_image(_unknown_event(store))
    past = time.time() - 2 * 3600
    os.utime(old_orphan, (past, past))
    os.utime(linked, (past, past))

    store.cleanup()

    assert not old_orphan.exists()
    assert new_orphan.exists()
    assert linked.exists()


# --- Открытие двери: миграция, разрешения, режим, привязки ---

def test_migrates_database_from_first_schema(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE persons (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, created_at REAL NOT NULL);
        CREATE TABLE events (id INTEGER PRIMARY KEY, camera_id INTEGER NOT NULL, started_at REAL NOT NULL,
            ended_at REAL NOT NULL, person_id INTEGER, image TEXT NOT NULL, embedding BLOB);
        INSERT INTO persons(name, created_at) VALUES ('Кирилл', 0);
        INSERT INTO events(camera_id, started_at, ended_at, person_id, image) VALUES (1, 0, 1, 1, 'a.jpg');
    """)
    conn.close()

    s = Store(db, tmp_path / "faces")
    try:
        person = s.list_persons()[0]
        assert person["name"] == "Кирилл" and person["can_unlock"] is False
        assert s.list_events()[0][0]["unlock"] is None
        assert s.unlock_mode() == "dry_run"
    finally:
        s.close()
    # Повторное открытие уже мигрированной БД — без ошибок
    Store(db, tmp_path / "faces").close()


def test_can_unlock_flag(store):
    person_id = store.create_person("Кирилл")
    assert store.can_unlock(person_id) is False
    store.set_can_unlock(person_id, True)
    assert store.can_unlock(person_id) is True
    assert store.list_persons()[0]["can_unlock"] is True
    assert store.can_unlock(999) is False
    with pytest.raises(NotFound):
        store.set_can_unlock(999, True)


def test_unlock_mode(store):
    assert store.unlock_mode() == "dry_run"
    store.set_unlock_mode("on")
    assert store.unlock_mode() == "on"
    with pytest.raises(Invalid):
        store.set_unlock_mode("always")


def test_camera_doors(store):
    assert store.camera_doors() == {}
    store.set_camera_door(1001, ("aabbccddeeff", 1, "Дверь подъезда"))
    store.set_camera_door(1001, ("aabbccddeeff", 2, "Калитка"))
    assert store.camera_doors() == {
        1001: {"id": "aabbccddeeff:2", "adapter_id": "aabbccddeeff", "relay": 2, "name": "Калитка"},
    }
    store.set_camera_door(1001, None)
    assert store.camera_doors() == {}


def test_unlock_result_on_event_and_filter(store):
    person_id = store.create_person("Кирилл")
    store.add_event(1001, 0, 1, person_id, b"j", None, unlock="would_open")
    store.add_event(1001, 2, 3, person_id, b"j", None)
    events, _ = store.list_events(unlocked=True)
    assert [e["unlock"] for e in events] == ["would_open"]


def test_ui_settings_default_set_and_reset(store):
    assert store.monitor_cameras() is None and store.match_threshold() is None
    store.set_monitor_cameras([12034, 1001, 12034])
    store.set_match_threshold(0.45)
    assert store.monitor_cameras() == [1001, 12034] and store.match_threshold() == 0.45
    store.set_monitor_cameras([])  # пустой набор — не то же, что «не задан»
    assert store.monitor_cameras() == []
    store.set_monitor_cameras(None)
    store.set_match_threshold(None)
    assert store.monitor_cameras() is None and store.match_threshold() is None


def test_ui_settings_survive_reopen(tmp_path):
    s = Store(tmp_path / "faces.db", tmp_path / "faces")
    s.set_monitor_cameras([12034])
    s.set_match_threshold(0.4)
    s.close()
    s = Store(tmp_path / "faces.db", tmp_path / "faces")
    assert s.monitor_cameras() == [12034] and s.match_threshold() == 0.4
    s.close()


@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan")])
def test_ui_settings_reject_bad_threshold(store, threshold):
    store.set_match_threshold(0.5)
    with pytest.raises(Invalid):
        store.set_match_threshold(threshold)
    assert store.match_threshold() == 0.5


def test_ui_settings_reject_bad_camera_ids(store):
    with pytest.raises(Invalid):
        store.set_monitor_cameras([True])
    assert store.monitor_cameras() is None
