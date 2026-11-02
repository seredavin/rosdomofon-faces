"""Вход в веб-интерфейс по паролю: подписанная cookie сессии, лимит попыток, middleware.

Cookie — "<expires>.<hmac>", ключ подписи — HMAC(секрет из файла, пароль): сессии
переживают перезапуск, а смена пароля делает недействительными все выданные.
"""

import hashlib
import hmac
import os
import secrets
import threading
import time
from collections import deque
from pathlib import Path

from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse, RedirectResponse

from . import config

COOKIE_NAME = "rdf_session"
# Доступны без входа; всё остальное — только с действующей сессией
PUBLIC_PATHS = frozenset({
    "/login", "/api/session", "/api/session/login", "/api/session/logout", "/api/health",
})
LOGIN_REQUIRED = {"detail": "Нужен вход в интерфейс", "code": "ui_login_required"}


class WrongPassword(Exception):
    pass


class TooManyAttempts(Exception):
    pass


def _load_secret(path: Path) -> bytes:
    """Читает секрет подписи, при отсутствии создаёт (32 байта, права 0600)."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(secret)
    return secret


class Auth:
    def __init__(
        self,
        password: str | None = None,
        secret_file: Path | None = None,
        max_age: float = config.SESSION_MAX_AGE,
        max_failures: int = config.LOGIN_MAX_FAILURES,
        failure_window: float = config.LOGIN_FAILURE_WINDOW,
    ):
        self.password = config.UI_PASSWORD if password is None else password
        self.secret_file = secret_file or config.SESSION_SECRET_FILE
        self.max_age = max_age
        self.max_failures = max_failures
        self.failure_window = failure_window
        self._key: bytes | None = None
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.password)

    def _sign(self, payload: str) -> str:
        if self._key is None:
            secret = _load_secret(self.secret_file)
            self._key = hmac.new(secret, self.password.encode(), hashlib.sha256).digest()
        return hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()

    def issue(self, now: float | None = None) -> str:
        expires = str(int((now or time.time()) + self.max_age))
        return f"{expires}.{self._sign(expires)}"

    def verify(self, cookie: str | None, now: float | None = None) -> bool:
        if not cookie or "." not in cookie:
            return False
        expires, sig = cookie.split(".", 1)
        if not expires.isdigit() or int(expires) <= (now or time.time()):
            return False
        return hmac.compare_digest(sig, self._sign(expires))

    def check_password(self, password: str) -> bool:
        # Сравниваем хеши: постоянное время независимо от длины пароля
        given = hashlib.sha256(password.encode()).digest()
        expected = hashlib.sha256(self.password.encode()).digest()
        return hmac.compare_digest(given, expected)

    def login(self, password: str, client: str, now: float | None = None) -> str:
        """Возвращает значение cookie или бросает TooManyAttempts / WrongPassword."""
        now = now or time.time()
        with self._lock:
            self._forget_old(now)
            attempts = self._failures.get(client)
            if attempts and len(attempts) >= self.max_failures:
                raise TooManyAttempts
            if not self.check_password(password):
                self._failures.setdefault(client, deque()).append(now)
                raise WrongPassword
            self._failures.pop(client, None)
        return self.issue(now)

    def _forget_old(self, now: float) -> None:
        for client in list(self._failures):
            attempts = self._failures[client]
            while attempts and attempts[0] <= now - self.failure_window:
                attempts.popleft()
            if not attempts:
                del self._failures[client]


class SessionMiddleware:
    """Чистый ASGI (не BaseHTTPMiddleware) — не оборачивает долгий MJPEG-поток.

    get_auth вызывается на каждый запрос, чтобы тесты могли подменить Auth.
    """

    def __init__(self, app, get_auth):
        self.app = app
        self.get_auth = get_auth

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        auth = self.get_auth()
        path = scope["path"]
        if not auth.enabled or path in PUBLIC_PATHS:
            return await self.app(scope, receive, send)
        if auth.verify(HTTPConnection(scope).cookies.get(COOKIE_NAME)):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        if path.startswith("/api/"):
            response = JSONResponse(LOGIN_REQUIRED, status_code=401)
        else:
            response = RedirectResponse("/login", status_code=303)
        await response(scope, receive, send)
