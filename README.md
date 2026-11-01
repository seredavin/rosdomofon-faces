# Росдомофон: поиск лиц

Веб-сервис: вход в аккаунт Росдомофона (SMS или refresh_token), просмотр камер
домофона в браузере, поиск лиц на видео (YuNet / OpenCV).

## Запуск

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m app            # http://127.0.0.1:8000
```

Модель детекции скачивается при первом запуске в `models/`.
Токены хранятся в `data/token.json` (права 0600) — не коммитить.

Сервер по умолчанию слушает только localhost: своей авторизации у него пока нет,
а через него доступны камеры и токены аккаунта.

## API

| Метод | Путь | Что делает |
|---|---|---|
| GET  | `/api/auth/status` | вошли ли мы |
| POST | `/api/auth/sms` `{phone}` | отправить SMS-код |
| POST | `/api/auth/code` `{phone, code}` | войти по коду |
| POST | `/api/auth/refresh-token` `{refresh_token}` | войти по refresh_token |
| POST | `/api/auth/logout` | выйти, удалить токены |
| GET  | `/api/cameras` | список камер |
| GET  | `/api/cameras/{id}/mjpeg` | видео с рамками вокруг лиц (MJPEG) |
| GET  | `/api/cameras/{id}/faces` | найденные лица в текущем кадре (JSON) |

## Устройство

- `app/rosdomofon.py` — клиент API, автообновление токена.
- `app/stream.py` — воркер на камеру: PyAV читает HLS с `Authorization: Bearer`,
  детектор ищет лица (не чаще `DETECT_MAX_FPS`), кадр кодируется в JPEG.
  Воркер стартует по первому зрителю и останавливается через 30 с без зрителей.
- `app/detector.py` — YuNet; `Face.landmarks` (глаза, нос, рот) пригодятся для
  выравнивания лица при распознавании.

На macOS при старте печатается предупреждение `objc: Class AVFFrameReceiver is
implemented in both ...` — OpenCV и PyAV везут свои копии ffmpeg. Дублируются
только классы захвата с веб-камеры, которые не используются; безвредно.
