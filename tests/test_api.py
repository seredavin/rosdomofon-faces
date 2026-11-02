"""Тесты HTTP API людей, образцов и журнала (без камер и без входа)."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.store import Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = Store(tmp_path / "faces.db", tmp_path / "faces")
    monkeypatch.setattr(main, "store", s)
    yield s
    s.close()


@pytest.fixture
def api(store):
    # Без контекстного менеджера lifespan не запускается: ни моделей, ни мониторинга
    return TestClient(main.app)


def _unknown_event(store, camera_id=1001) -> int:
    t = time.time()
    return store.add_event(camera_id, t - 3, t, None, b"jpeg-bytes", np.ones(128, np.float32))


def test_gallery_assign_to_new_person(api, store):
    event_id = _unknown_event(store)
    gallery = api.get("/api/events", params={"gallery": True}).json()
    assert [e["id"] for e in gallery["items"]] == [event_id]

    resp = api.post(f"/api/events/{event_id}/assign", json={"name": "Кирилл"})
    assert resp.status_code == 200
    person_id = resp.json()["person_id"]

    assert api.get("/api/events", params={"gallery": True}).json()["items"] == []
    persons = api.get("/api/persons").json()
    assert persons[0]["id"] == person_id and len(persons[0]["samples"]) == 1
    sample_id = persons[0]["samples"][0]["id"]
    image = api.get(f"/api/samples/{sample_id}/image")
    assert image.status_code == 200 and image.content == b"jpeg-bytes"
    assert image.headers["content-type"] == "image/jpeg"


def test_assign_errors(api, store):
    event_id = _unknown_event(store)
    api.post("/api/persons", json={"name": "Кирилл"})
    assert api.post(f"/api/events/{event_id}/assign", json={"name": "Кирилл"}).status_code == 409
    assert api.post(f"/api/events/{event_id}/assign", json={}).status_code == 400
    assert api.post(f"/api/events/{event_id}/assign", json={"person_id": 999}).status_code == 404
    assert api.post("/api/events/999/assign", json={"name": "Аня"}).status_code == 404


def test_person_crud(api):
    assert api.post("/api/persons", json={"name": "  "}).status_code == 400
    person_id = api.post("/api/persons", json={"name": "Аня"}).json()["id"]
    assert api.post("/api/persons", json={"name": "Аня"}).status_code == 409
    assert api.patch(f"/api/persons/{person_id}", json={"name": "Анна"}).status_code == 200
    assert api.get("/api/persons").json()[0]["name"] == "Анна"
    assert api.delete(f"/api/persons/{person_id}").status_code == 200
    assert api.delete(f"/api/persons/{person_id}").status_code == 404


def test_delete_sample(api, store):
    store.assign_event(_unknown_event(store), name="Кирилл")
    sample_id = store.list_persons()[0]["samples"][0]["id"]
    assert api.delete(f"/api/samples/{sample_id}").status_code == 200
    assert api.get(f"/api/samples/{sample_id}/image").status_code == 404


def test_events_filters_pagination_and_delete(api, store):
    ids = [_unknown_event(store, camera_id=1001 if i % 2 else 1002) for i in range(5)]
    page = api.get("/api/events", params={"limit": 2}).json()
    assert [e["id"] for e in page["items"]] == ids[::-1][:2]
    page2 = api.get("/api/events", params={"limit": 2, "before_id": page["next_before_id"]}).json()
    assert [e["id"] for e in page2["items"]] == ids[::-1][2:4]
    only = api.get("/api/events", params={"camera_id": 1002}).json()["items"]
    assert {e["camera_id"] for e in only} == {1002}

    assert api.get(f"/api/events/{ids[0]}/image").content == b"jpeg-bytes"
    assert api.delete(f"/api/events/{ids[0]}").status_code == 200
    assert api.get(f"/api/events/{ids[0]}/image").status_code == 404
    assert api.get("/api/events", params={"limit": 500}).status_code == 422


def test_status_without_login(api, monkeypatch):
    monkeypatch.setattr(main.client, "_tok", None)
    resp = api.get("/api/status").json()
    assert resp["authorized"] is False
    assert api.get("/api/cameras").status_code == 401


# --- Открытие двери ---

from app.rosdomofon import Door  # noqa: E402

DOORS = [Door("aabbccddeeff", 1, "Дверь подъезда"), Door("aabbccddeeff", 2, "Калитка")]


@pytest.fixture
def doors(monkeypatch):
    monkeypatch.setattr(main.client, "doors", lambda: DOORS)


def test_doors_and_unlock_settings(api, doors):
    assert api.get("/api/doors").json() == [
        {"id": "aabbccddeeff:1", "name": "Дверь подъезда"},
        {"id": "aabbccddeeff:2", "name": "Калитка"},
    ]
    assert api.get("/api/unlock").json() == {"mode": "dry_run", "bindings": {}}

    resp = api.put("/api/unlock", json={"mode": "on", "bindings": {"1001": "aabbccddeeff:2"}})
    assert resp.json() == {"mode": "on", "bindings": {"1001": {"id": "aabbccddeeff:2", "name": "Калитка"}}}

    resp = api.put("/api/unlock", json={"bindings": {"1001": None}})
    assert resp.json() == {"mode": "on", "bindings": {}}


def test_unlock_settings_errors(api, doors):
    assert api.put("/api/unlock", json={"mode": "always"}).status_code == 400
    # Неверная дверь — запрос отклонён целиком, режим тоже не меняется
    assert api.put("/api/unlock", json={"mode": "on", "bindings": {"1001": "nope:1"}}).status_code == 400
    assert api.get("/api/unlock").json()["mode"] == "dry_run"


def test_person_can_unlock_flag(api):
    person_id = api.post("/api/persons", json={"name": "Кирилл"}).json()["id"]
    assert api.get("/api/persons").json()[0]["can_unlock"] is False
    assert api.patch(f"/api/persons/{person_id}", json={"can_unlock": True}).status_code == 200
    person = api.get("/api/persons").json()[0]
    assert person["can_unlock"] is True and person["name"] == "Кирилл"
    assert api.patch("/api/persons/999", json={"can_unlock": True}).status_code == 404


def test_events_unlocked_filter(api, store):
    person_id = store.create_person("Кирилл")
    store.add_event(1001, 0, 1, person_id, b"j", None, unlock="opened")
    store.add_event(1001, 2, 3, person_id, b"j", None)
    items = api.get("/api/events", params={"unlocked": True}).json()["items"]
    assert [e["unlock"] for e in items] == ["opened"]
