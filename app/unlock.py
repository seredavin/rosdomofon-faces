"""Автоматическое открытие двери для знакомых людей с разрешением."""

import logging
import threading
import time

from . import config
from .rosdomofon import AuthError, DoorError, RosdomofonClient
from .store import UNLOCK_DRY_RUN, UNLOCK_OFF, Store

_LOGGER = logging.getLogger(__name__)

WOULD_OPEN = "would_open"
OPENED = "opened"
FAILED = "failed"


class DoorUnlocker:
    """Один на сервис: общие для всех камер паузы по дверям. Потокобезопасен."""

    def __init__(self, client: RosdomofonClient, store: Store, clock=time.monotonic) -> None:
        self._client = client
        self._store = store
        self._clock = clock
        self._last_open: dict[str, float] = {}
        self._lock = threading.Lock()

    def try_unlock(self, camera_id: int, person_id: int) -> str | None:
        """Открывает дверь камеры для человека, если это положено.

        Возвращает None, если открытие не положено (режим выключен, камера не
        привязана, нет разрешения, дверь в паузе), иначе результат для журнала.
        """
        mode = self._store.unlock_mode()
        if mode == UNLOCK_OFF:
            return None
        door = self._store.camera_doors().get(camera_id)
        if door is None or not self._store.can_unlock(person_id):
            return None
        if not self._reserve(door["id"]):
            _LOGGER.info("Камера %s: «%s» в паузе после открытия", camera_id, door["name"])
            return None

        if mode == UNLOCK_DRY_RUN:
            _LOGGER.info("Камера %s: открыл бы «%s» для человека %s", camera_id, door["name"], person_id)
            return WOULD_OPEN
        try:
            self._client.open_door(door["adapter_id"], door["relay"])
        except (DoorError, AuthError) as exc:
            _LOGGER.error("Камера %s: не удалось открыть «%s»: %s", camera_id, door["name"], exc)
            return FAILED
        _LOGGER.warning("Камера %s: открыта «%s» для человека %s", camera_id, door["name"], person_id)
        return OPENED

    def _reserve(self, door_id: str) -> bool:
        """Занимает дверь на UNLOCK_COOLDOWN секунд; False, если она ещё в паузе."""
        now = self._clock()
        with self._lock:
            last = self._last_open.get(door_id)
            if last is not None and now - last < config.UNLOCK_COOLDOWN:
                return False
            self._last_open[door_id] = now
            return True
