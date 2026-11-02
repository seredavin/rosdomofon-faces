"""Веб-сервис: вход в Росдомофон, камеры, распознавание лиц, люди и журнал событий."""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import config
from .detector import ensure_models
from .rosdomofon import AuthError, RosdomofonClient, normalize_phone
from .store import Conflict, Invalid, NotFound, Store, StoreError
from .stream import WorkerRegistry
from .unlock import DoorUnlocker

_LOGGER = logging.getLogger(__name__)

client = RosdomofonClient()
store = Store()
unlocker = DoorUnlocker(client, store)
workers = WorkerRegistry(client, store, unlocker)


def _start_monitoring() -> None:
    """Запускает мониторинг; ошибки только логируются — вход и сервис работают и без него."""
    try:
        workers.start_monitoring()
    except (AuthError, httpx.HTTPError) as exc:
        _LOGGER.warning("Не удалось запустить мониторинг камер: %s", exc)


def _cleanup_loop(stop: threading.Event) -> None:
    """Очистка по срокам хранения: сразу при старте и затем раз в CLEANUP_INTERVAL."""
    while not stop.is_set():
        try:
            store.cleanup()
        except Exception:  # noqa: BLE001 — очистка не должна останавливать сервис
            _LOGGER.exception("Ошибка очистки")
        stop.wait(config.CLEANUP_INTERVAL)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(ensure_models)
    stop_cleanup = threading.Event()
    threading.Thread(target=_cleanup_loop, args=(stop_cleanup,), name="cleanup", daemon=True).start()
    if client.is_authorized:
        await asyncio.to_thread(_start_monitoring)
    yield
    stop_cleanup.set()
    await asyncio.to_thread(workers.stop_all)


app = FastAPI(title="Росдомофон: лица", lifespan=lifespan)


class PhoneIn(BaseModel):
    phone: str


class CodeIn(BaseModel):
    phone: str
    code: str


class RefreshTokenIn(BaseModel):
    refresh_token: str


class PersonIn(BaseModel):
    name: str


class AssignIn(BaseModel):
    person_id: int | None = None
    name: str | None = None


def _phone(raw: str) -> str:
    try:
        return normalize_phone(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _call(fn, *args, **kwargs):
    """Выполняет вызов, превращая ошибки API Росдомофона и хранилища в HTTP-ответы."""
    try:
        return fn(*args, **kwargs)
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Росдомофон недоступен: {exc}") from exc
    except NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except (Invalid, StoreError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _image(path) -> FileResponse:
    if not path.exists():
        raise HTTPException(404, "Снимок не найден")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


# --- Авторизация ---

@app.get("/api/auth/status")
def auth_status():
    return {"authorized": client.is_authorized}


@app.post("/api/auth/sms")
def auth_sms(body: PhoneIn):
    _call(client.request_sms, _phone(body.phone))
    return {"ok": True}


@app.post("/api/auth/code")
def auth_code(body: CodeIn):
    _call(client.login_sms, _phone(body.phone), body.code.strip())
    _start_monitoring()
    return {"ok": True}


@app.post("/api/auth/refresh-token")
def auth_refresh_token(body: RefreshTokenIn):
    _call(client.login_refresh_token, body.refresh_token.strip())
    _start_monitoring()
    return {"ok": True}


@app.post("/api/auth/logout")
def auth_logout():
    workers.stop_all()
    client.logout()
    return {"ok": True}


# --- Камеры ---

def _camera(camera_id: int):
    camera = _call(client.camera, camera_id)
    if camera is None:
        raise HTTPException(404, "Камера не найдена")
    return camera


@app.get("/api/cameras")
def cameras():
    return [{"id": c.id, "name": c.name} for c in _call(client.cameras)]


@app.get("/api/status")
def status():
    return {"authorized": client.is_authorized, "cameras": workers.statuses()}


@app.get("/api/cameras/{camera_id}/mjpeg")
def camera_mjpeg(camera_id: int):
    worker = workers.get(_camera(camera_id))

    def frames():
        frame_no = 0
        while not worker.stopped:
            frame_no, jpeg = worker.wait_frame(frame_no)
            if jpeg is None:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                + jpeg + b"\r\n"
            )

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/cameras/{camera_id}/faces")
def camera_faces(camera_id: int):
    return workers.get(_camera(camera_id)).faces()


# --- Журнал событий и галерея незнакомых ---

@app.get("/api/events")
def events(
    camera_id: int | None = None,
    person_id: int | None = None,
    unknown: bool = False,
    gallery: bool = False,
    since: float | None = None,
    until: float | None = None,
    before_id: int | None = None,
    limit: int = Query(50, ge=1, le=200),
):
    items, next_before_id = store.list_events(
        camera_id=camera_id, person_id=person_id, unknown=unknown, gallery=gallery,
        since=since, until=until, before_id=before_id, limit=limit,
    )
    return {"items": items, "next_before_id": next_before_id}


@app.get("/api/events/{event_id}/image")
def event_image(event_id: int):
    return _image(_call(store.event_image, event_id))


@app.delete("/api/events/{event_id}")
def delete_event(event_id: int):
    _call(store.delete_event, event_id)
    return {"ok": True}


@app.post("/api/events/{event_id}/assign")
def assign_event(event_id: int, body: AssignIn):
    person_id = _call(store.assign_event, event_id, person_id=body.person_id, name=body.name)
    return {"ok": True, "person_id": person_id}


# --- Люди и образцы ---

@app.get("/api/persons")
def persons():
    return store.list_persons()


@app.post("/api/persons")
def create_person(body: PersonIn):
    return {"id": _call(store.create_person, body.name)}


@app.patch("/api/persons/{person_id}")
def rename_person(person_id: int, body: PersonIn):
    _call(store.rename_person, person_id, body.name)
    return {"ok": True}


@app.delete("/api/persons/{person_id}")
def delete_person(person_id: int):
    _call(store.delete_person, person_id)
    return {"ok": True}


@app.delete("/api/samples/{sample_id}")
def delete_sample(sample_id: int):
    _call(store.delete_sample, sample_id)
    return {"ok": True}


@app.get("/api/samples/{sample_id}/image")
def sample_image(sample_id: int):
    return _image(_call(store.sample_image, sample_id))


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")
