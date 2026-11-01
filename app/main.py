"""Веб-сервис: вход в Росдомофон, список камер, видео с найденными лицами."""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import config
from .detector import ensure_model
from .rosdomofon import AuthError, RosdomofonClient, normalize_phone
from .stream import WorkerRegistry

_LOGGER = logging.getLogger(__name__)

client = RosdomofonClient()
workers = WorkerRegistry(client)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_model()
    yield
    workers.stop_all()


app = FastAPI(title="Росдомофон: лица", lifespan=lifespan)


class PhoneIn(BaseModel):
    phone: str


class CodeIn(BaseModel):
    phone: str
    code: str


class RefreshTokenIn(BaseModel):
    refresh_token: str


def _phone(raw: str) -> str:
    try:
        return normalize_phone(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _auth_call(fn, *args) -> dict:
    """Выполняет вызов клиента, превращая ошибки API в понятные HTTP-ответы."""
    try:
        fn(*args)
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Росдомофон недоступен: {exc}") from exc
    return {"ok": True}


# --- Авторизация ---

@app.get("/api/auth/status")
def auth_status():
    return {"authorized": client.is_authorized}


@app.post("/api/auth/sms")
def auth_sms(body: PhoneIn):
    return _auth_call(client.request_sms, _phone(body.phone))


@app.post("/api/auth/code")
def auth_code(body: CodeIn):
    return _auth_call(client.login_sms, _phone(body.phone), body.code.strip())


@app.post("/api/auth/refresh-token")
def auth_refresh_token(body: RefreshTokenIn):
    return _auth_call(client.login_refresh_token, body.refresh_token.strip())


@app.post("/api/auth/logout")
def auth_logout():
    workers.stop_all()
    client.logout()
    return {"ok": True}


# --- Камеры ---

def _camera(camera_id: int):
    try:
        camera = client.camera(camera_id)
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Росдомофон недоступен: {exc}") from exc
    if camera is None:
        raise HTTPException(404, "Камера не найдена")
    return camera


@app.get("/api/cameras")
def cameras():
    try:
        return [{"id": c.id, "name": c.name} for c in client.cameras()]
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Росдомофон недоступен: {exc}") from exc


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
    return workers.get(_camera(camera_id)).status()


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")
