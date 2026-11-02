"""Тесты входа в веб-интерфейс: сессии, защита API, лимит попыток."""

import os
import stat
import time

import pytest
from fastapi.testclient import TestClient

from app import main
from app.session import COOKIE_NAME, Auth, TooManyAttempts, WrongPassword

PASSWORD = "correct horse battery staple"


@pytest.fixture
def secret_file(tmp_path):
    return tmp_path / "session_secret"


@pytest.fixture
def auth(secret_file, monkeypatch):
    a = Auth(PASSWORD, secret_file)
    monkeypatch.setattr(main, "auth", a)
    return a


@pytest.fixture
def api(auth):
    return TestClient(main.app)


def _login(api, password=PASSWORD):
    return api.post("/api/session/login", json={"password": password})


def test_disabled_without_password(secret_file, monkeypatch):
    monkeypatch.setattr(main, "auth", Auth("", secret_file))
    api = TestClient(main.app)
    assert api.get("/api/persons").status_code == 200
    assert api.get("/api/session").json() == {"enabled": False, "authenticated": True}
    assert api.get("/login", follow_redirects=False).status_code == 303
    assert not secret_file.exists()


@pytest.mark.parametrize("path", [
    "/api/persons", "/api/events", "/api/auth/status", "/api/cameras/1001/mjpeg",
    "/api/events/1/image", "/api/samples/1/image", "/api/unlock",
])
def test_api_requires_session(api, path):
    resp = api.get(path)
    assert resp.status_code == 401
    assert resp.json()["code"] == "ui_login_required"


def test_mutations_require_session(api):
    assert api.post("/api/persons", json={"name": "Аня"}).status_code == 401
    assert api.put("/api/unlock", json={"mode": "on"}).status_code == 401


def test_pages_redirect_to_login(api):
    resp = api.get("/", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/login"
    login = api.get("/login")
    assert login.status_code == 200 and 'type="password"' in login.text


def test_public_endpoints(api):
    assert api.get("/api/health").json() == {"ok": True}
    assert api.get("/api/session").json() == {"enabled": True, "authenticated": False}


def test_login_logout(api, secret_file):
    resp = _login(api)
    assert resp.status_code == 200
    cookie = resp.headers["set-cookie"]
    assert COOKIE_NAME in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert "Secure" not in cookie  # запрос по http
    assert stat.S_IMODE(os.stat(secret_file).st_mode) == 0o600

    assert api.get("/api/session").json()["authenticated"] is True
    assert api.get("/api/persons").status_code == 200
    assert api.get("/", follow_redirects=False).status_code == 200
    assert api.get("/login", follow_redirects=False).status_code == 303

    assert api.post("/api/session/logout").status_code == 200
    assert api.get("/api/persons").status_code == 401


def test_secure_cookie_over_https(auth):
    api = TestClient(main.app, base_url="https://testserver")
    assert "Secure" in _login(api).headers["set-cookie"]


def test_wrong_password(api):
    resp = _login(api, "wrong")
    assert resp.status_code == 401 and resp.json()["detail"] == "Неверный пароль"
    assert "set-cookie" not in resp.headers
    assert api.get("/api/persons").status_code == 401


def test_cookie_validation(auth, secret_file):
    now = time.time()
    cookie = auth.issue(now)
    assert auth.verify(cookie, now)
    # Срок истёк
    assert not auth.verify(cookie, now + auth.max_age + 1)
    # Подделка подписи и срока
    expires, sig = cookie.split(".")
    assert not auth.verify(f"{expires}.{'0' * len(sig)}", now)
    assert not auth.verify(f"{int(expires) + 1000}.{sig}", now)
    for junk in (None, "", "abc", "x.y", ".", "-1.abc"):
        assert not auth.verify(junk, now)
    # Перезапуск с тем же паролем и секретом — сессия действует
    assert Auth(PASSWORD, secret_file).verify(cookie, now)
    # Смена пароля — все сессии недействительны
    assert not Auth("new password", secret_file).verify(cookie, now)


def test_expired_cookie_rejected_by_api(api, auth):
    api.cookies.set(COOKIE_NAME, auth.issue(time.time() - auth.max_age - 1))
    assert api.get("/api/persons").status_code == 401


def test_rate_limit(api):
    for _ in range(5):
        assert _login(api, "wrong").status_code == 401
    resp = _login(api)
    assert resp.status_code == 429
    assert resp.json()["detail"] == "Слишком много попыток, попробуйте позже"
    assert api.get("/api/persons").status_code == 401


def test_rate_limit_window_and_reset(secret_file):
    auth = Auth(PASSWORD, secret_file, max_failures=3, failure_window=60)
    t = 1000.0
    for _ in range(2):
        with pytest.raises(WrongPassword):
            auth.login("wrong", "1.2.3.4", t)
    # Успех до порога обнуляет счётчик
    auth.login(PASSWORD, "1.2.3.4", t)
    for _ in range(3):
        with pytest.raises(WrongPassword):
            auth.login("wrong", "1.2.3.4", t)
    with pytest.raises(TooManyAttempts):
        auth.login(PASSWORD, "1.2.3.4", t + 1)
    # Другой адрес не заблокирован
    auth.login(PASSWORD, "5.6.7.8", t + 1)
    # После окна — снова можно
    auth.login(PASSWORD, "1.2.3.4", t + 61)
