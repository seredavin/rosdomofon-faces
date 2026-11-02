## 1. Настройки

- [x] 1.1 `app/config.py`: `UI_PASSWORD` (`ROSDOMOFON_UI_PASSWORD`, пусто — выключено), `SESSION_SECRET_FILE = DATA_DIR / "session_secret"`, `SESSION_MAX_AGE` (30 дней), `LOGIN_MAX_FAILURES = 5`, `LOGIN_FAILURE_WINDOW` (15 минут), `FORWARDED_ALLOW_IPS` (`ROSDOMOFON_FORWARDED_ALLOW_IPS`, по умолчанию `127.0.0.1`)
- [x] 1.2 `app/__main__.py`: передать `proxy_headers=True` и `forwarded_allow_ips` в `uvicorn.run`; обновить комментарий про отсутствие авторизации

## 2. Сессии и лимит попыток

- [x] 2.1 `app/session.py`: загрузка/создание `session_secret` (32 байта, права 0600)
- [x] 2.2 Проверка пароля через `hmac.compare_digest` от SHA-256; выдача и проверка cookie `<expires>.<hmac>` с ключом от секрета и пароля
- [x] 2.3 Лимит неудачных попыток по IP (скользящее окно, сброс при успехе, вычистка старых записей)
- [x] 2.4 Чистый ASGI-middleware: белый список публичных путей, 401 `{"code": "ui_login_required"}` для `/api/*`, 303 на `/login` для остального; при выключенной авторизации пропускает всё

## 3. API

- [x] 3.1 `app/main.py`: подключить middleware; `WARNING` в лог при старте, если пароль не задан
- [x] 3.2 `GET /api/session`, `POST /api/session/login` (200 + cookie / 401 «Неверный пароль» / 429 «Слишком много попыток, попробуйте позже»), `POST /api/session/logout` (удаление cookie)
- [x] 3.3 `GET /api/health` → `{"ok": true}`; `GET /login` → `login.html` (вошедшего — 303 на `/`)

## 4. Интерфейс

- [x] 4.1 `app/static/login.html`: форма пароля, ошибки 401/429, при успехе переход на `/`
- [x] 4.2 `index.html`: в `api()` при `code === "ui_login_required"` — переход на `/login`; кнопка «Выйти из интерфейса» (видна, только если `enabled`), отдельная от выхода из Росдомофона

## 5. Тесты

- [x] 5.1 `tests/test_session.py`: без пароля всё открыто; с паролем — 401 для API (включая MJPEG и снимки), 303 для `/`, публичны `/login` и `/api/health`
- [x] 5.2 Вход верным/неверным паролем, выход, истёкшая cookie, подделанная cookie, смена пароля инвалидирует cookie, сессия переживает пересоздание (тот же секрет)
- [x] 5.3 Лимит: 6-я неудачная попытка → 429 даже с верным паролем; успех до порога сбрасывает счётчик
- [x] 5.4 Прогнать весь `pytest`: существующие тесты API проходят без пароля

## 6. Docker и документация

- [x] 6.1 `Dockerfile`: healthcheck на `/api/health`, обновить комментарий у `CMD`
- [x] 6.2 `README.md`: раздел «Публикация в интернет» (пароль `openssl rand -base64 24`, reverse proxy с HTTPS, пример Caddy, `ROSDOMOFON_FORWARDED_ALLOW_IPS`), новые переменные в таблице настроек, `/api/session/*` и `/api/health` в разделе API, обновить предупреждения про localhost
- [x] 6.3 `openspec/config.yaml`: обновить контекст («своей авторизации нет»)
- [x] 6.4 Проверить вручную в браузере: вход, видео и снимки после входа, выход, переход на `/login` при истёкшей сессии
