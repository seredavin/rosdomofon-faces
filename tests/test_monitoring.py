"""Тесты набора камер мониторинга в реестре воркеров (без сети и видео)."""

import pytest

from app import config, stream
from app.rosdomofon import Camera
from app.store import Store

CAMERAS = [Camera(1001, "Подъезд", "s.example"), Camera(12034, "Калитка", "s.example")]


class FakeClient:
    def cameras(self):
        return CAMERAS


class FakeWorker:
    """Вместо потока камеры: запоминает флаги, ничего не читает."""

    def __init__(self, _client, _store, camera, persistent=False, _unlocker=None):
        self.camera = camera
        self.persistent = persistent
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def touch(self):
        pass

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        pass


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "faces.db", tmp_path / "faces")
    yield s
    s.close()


@pytest.fixture
def registry(store, monkeypatch):
    monkeypatch.setattr(stream, "CameraWorker", FakeWorker)
    monkeypatch.setattr(config, "MONITOR_CAMERAS", None)
    return stream.WorkerRegistry(FakeClient(), store)


def _monitored(registry) -> set[int]:
    return {w.camera.id for w in registry._workers.values() if w.persistent}


def test_default_from_env_all_cameras(registry):
    registry.sync_monitoring()
    assert _monitored(registry) == {1001, 12034}


def test_default_from_env_subset(registry, monkeypatch):
    monkeypatch.setattr(config, "MONITOR_CAMERAS", frozenset({1001}))
    registry.sync_monitoring()
    assert _monitored(registry) == {1001}


def test_ui_set_overrides_env(registry, store, monkeypatch):
    monkeypatch.setattr(config, "MONITOR_CAMERAS", frozenset({1001}))
    store.set_monitor_cameras([12034])
    registry.sync_monitoring()
    assert _monitored(registry) == {12034}


def test_add_and_remove_camera_without_restart(registry, store):
    store.set_monitor_cameras([1001])
    registry.sync_monitoring()
    first = registry._workers[1001]

    store.set_monitor_cameras([12034])
    registry.sync_monitoring()
    assert _monitored(registry) == {12034}
    assert registry._workers[12034].started
    # Исключённая камера не останавливается принудительно — уйдёт сама без зрителей
    assert first.persistent is False and first.stopped is False


def test_repeat_sync_does_not_duplicate_workers(registry):
    registry.sync_monitoring()
    workers = dict(registry._workers)
    registry.sync_monitoring()
    assert registry._workers == workers


def test_viewer_worker_becomes_persistent(registry, store):
    store.set_monitor_cameras([])
    viewer = registry.get(CAMERAS[1])
    assert viewer.persistent is False
    store.set_monitor_cameras([12034])
    registry.sync_monitoring()
    assert registry._workers[12034] is viewer and viewer.persistent is True


def test_empty_set_disables_monitoring(registry, store):
    registry.sync_monitoring()
    store.set_monitor_cameras([])
    registry.sync_monitoring()
    assert _monitored(registry) == set()


def test_real_worker_stops_after_leaving_monitoring(store, monkeypatch):
    """Настоящий воркер без флага мониторинга и без зрителей считается простаивающим."""
    import time

    worker = stream.CameraWorker(FakeClient(), store, CAMERAS[0], persistent=True)
    worker._last_view = time.monotonic() - config.WORKER_IDLE_TIMEOUT - 1
    assert worker._idle() is False
    worker.persistent = False
    assert worker._idle() is True
    worker.touch()  # камеру смотрят — работает дальше
    assert worker._idle() is False
