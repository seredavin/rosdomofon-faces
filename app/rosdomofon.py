"""Клиент API Росдомофона: вход по SMS / refresh_token, обновление токена, камеры."""

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass

import httpx

from . import config

_LOGGER = logging.getLogger(__name__)

# Запас (в секундах) до истечения токена, при котором запускается refresh
_EXPIRY_MARGIN = 60
_TIMEOUT = httpx.Timeout(10.0)


class AuthError(Exception):
    """Ошибка авторизации в Росдомофоне (нужно войти заново)."""


@dataclass(frozen=True)
class Camera:
    id: int
    name: str
    stream_host: str

    @property
    def hls_url(self) -> str:
        return f"https://{self.stream_host}/live/{self.id}.m3u8"


def normalize_phone(raw_phone: str) -> str:
    """Приводит номер к формату 7XXXXXXXXXX (как в HA-интеграции)."""
    digits = re.sub(r"\D", "", raw_phone)
    if digits.startswith("8") and len(digits) == 11:
        digits = "7" + digits[1:]
    if len(digits) != 11 or not digits.startswith("7"):
        raise ValueError("Номер должен быть российским: 11 цифр, начиная с 7 или 8")
    return digits


def _stream_host(rdva_uri: str) -> str:
    """rdvaUri из API → хост HLS-сервера (rdva1.rosdomofon.com → s.rdva1.rosdomofon.com)."""
    host = re.sub(r"^https?://", "", rdva_uri).strip("/")
    if not host.startswith("s."):
        host = f"s.{host}"
    if not host.endswith(".rosdomofon.com"):
        raise ValueError(f"Неожиданный хост видеосервера: {host}")
    return host


class RosdomofonClient:
    """Потокобезопасный клиент: токены хранятся в data/token.json и обновляются автоматически."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._http = httpx.Client(timeout=_TIMEOUT)
        self._tok: dict | None = self._load()
        self._cameras: dict[int, Camera] = {}

    # --- Авторизация ---

    @property
    def is_authorized(self) -> bool:
        return self._tok is not None

    def request_sms(self, phone: str) -> None:
        resp = self._http.post(
            config.SMS_REQUEST_URL.format(phone=phone),
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            raise AuthError(f"Не удалось отправить SMS: HTTP {resp.status_code}")

    def login_sms(self, phone: str, code: str) -> None:
        self._set_token(self._token_request({
            "grant_type": "mobile",
            "client_id": config.CLIENT_ID,
            "phone": phone,
            "sms_code": code,
            "company": "",
        }))

    def login_refresh_token(self, refresh_token: str) -> None:
        tok = self._token_request({
            "grant_type": "refresh_token",
            "client_id": config.CLIENT_ID,
            "refresh_token": refresh_token,
        })
        tok.setdefault("refresh_token", refresh_token)
        self._set_token(tok)

    def logout(self) -> None:
        with self._lock:
            self._tok = None
            self._cameras = {}
            config.TOKEN_FILE.unlink(missing_ok=True)

    def access_token(self, force_refresh: bool = False) -> str:
        """Возвращает действующий access_token, при необходимости обновляя его."""
        with self._lock:
            if self._tok is None:
                raise AuthError("Не выполнен вход")
            if force_refresh or self._is_expired():
                _LOGGER.info("Обновляю access_token")
                try:
                    tok = self._token_request({
                        "grant_type": "refresh_token",
                        "client_id": config.CLIENT_ID,
                        "refresh_token": self._tok["refresh_token"],
                    })
                except AuthError:
                    # refresh_token тоже недействителен — нужен повторный вход
                    self._tok = None
                    config.TOKEN_FILE.unlink(missing_ok=True)
                    raise
                tok.setdefault("refresh_token", self._tok["refresh_token"])
                self._store(tok)
            return self._tok["access_token"]

    # --- Камеры ---

    def cameras(self) -> list[Camera]:
        resp = self._authorized_get(config.CAMERAS_LIST_URL)
        cameras = {}
        for item in resp.json():
            if not item.get("id") or not item.get("rdvaUri"):
                continue
            location = (item.get("location") or {}).get("name")
            cameras[item["id"]] = Camera(
                id=item["id"],
                name=location or f"Камера {item['id']}",
                stream_host=_stream_host(item["rdvaUri"]),
            )
        self._cameras = cameras
        return list(cameras.values())

    def camera(self, camera_id: int) -> Camera | None:
        if camera_id not in self._cameras:
            self.cameras()
        return self._cameras.get(camera_id)

    # --- Внутреннее ---

    def _authorized_get(self, url: str) -> httpx.Response:
        resp = self._http.get(url, headers={"Authorization": f"Bearer {self.access_token()}"})
        if resp.status_code == 401:
            resp = self._http.get(
                url, headers={"Authorization": f"Bearer {self.access_token(force_refresh=True)}"}
            )
        resp.raise_for_status()
        return resp

    def _token_request(self, payload: dict) -> dict:
        resp = self._http.post(config.TOKEN_REQUEST_URL, data=payload)
        if resp.status_code != 200:
            _LOGGER.warning("oauth/token: HTTP %d %s", resp.status_code, resp.text[:200])
            raise AuthError(f"Сервер отклонил запрос токена: HTTP {resp.status_code}")
        tok = resp.json()
        if not tok.get("access_token") or "expires_in" not in tok:
            raise AuthError("В ответе сервера нет access_token/expires_in")
        return tok

    def _set_token(self, tok: dict) -> None:
        with self._lock:
            self._store(tok)

    def _store(self, tok: dict) -> None:
        tok["timestamp"] = int(time.time())
        self._tok = tok
        config.DATA_DIR.mkdir(exist_ok=True)
        fd = os.open(config.TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(tok, f)

    def _is_expired(self) -> bool:
        elapsed = time.time() - self._tok.get("timestamp", 0)
        return elapsed > self._tok["expires_in"] - _EXPIRY_MARGIN

    @staticmethod
    def _load() -> dict | None:
        try:
            return json.loads(config.TOKEN_FILE.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            _LOGGER.warning("Не удалось прочитать %s: %s", config.TOKEN_FILE, exc)
            return None
