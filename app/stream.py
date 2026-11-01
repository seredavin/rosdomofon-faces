"""Воркеры камер: чтение HLS с Bearer-токеном, поиск лиц, отдача JPEG-кадров."""

import logging
import queue
import threading
import time
from dataclasses import asdict

import av
import cv2

from . import config
from .detector import Face, FaceDetector, draw_faces
from .rosdomofon import AuthError, Camera, RosdomofonClient

_LOGGER = logging.getLogger(__name__)

# Пауза между попытками переподключения к потоку
_RECONNECT_DELAY = 3
# Расхождение (сек) между PTS кадра и часами, после которого отсчёт сбрасывается
_MAX_DRIFT = 5
# Сколько сжатых пакетов (≈ кадров) держать в очереди — около 40 с видео при 25 к/с
_PACKET_BUFFER = 1000


def _read_packets(container, stream, packets: queue.Queue, done: threading.Event) -> None:
    """Поток чтения: складывает сжатые пакеты в очередь, в конце — None или исключение."""
    item = None
    try:
        for packet in container.demux(stream):
            if packet.dts is None:  # служебный пустой пакет в конце потока
                continue
            if not _put(packets, packet, done):
                return
    except Exception as exc:  # noqa: BLE001 — пробрасывается в воркер через очередь
        item = exc
    _put(packets, item, done)


def _put(packets: queue.Queue, item, done: threading.Event) -> bool:
    """Кладёт элемент в очередь, пока воркер не попросил остановиться."""
    while not done.is_set():
        try:
            packets.put(item, timeout=1)
            return True
        except queue.Full:
            continue
    return False


def _drain(packets: queue.Queue) -> None:
    while True:
        try:
            packets.get_nowait()
        except queue.Empty:
            return


class CameraWorker(threading.Thread):
    """Фоновый поток одной камеры. Останавливается сам, если его долго никто не читает."""

    def __init__(self, client: RosdomofonClient, camera: Camera) -> None:
        super().__init__(name=f"camera-{camera.id}", daemon=True)
        self._client = client
        self.camera = camera
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_no = 0
        self._faces: list[Face] = []
        self._error: str | None = None
        self._last_access = time.monotonic()
        self._stopped = threading.Event()

    # --- Для HTTP-обработчиков ---

    def touch(self) -> None:
        self._last_access = time.monotonic()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def wait_frame(self, after: int, timeout: float = 10) -> tuple[int, bytes | None]:
        """Ждёт кадр новее `after`; возвращает (номер, JPEG)."""
        self.touch()
        with self._cond:
            self._cond.wait_for(lambda: self._frame_no > after or self.stopped, timeout)
            return self._frame_no, self._jpeg

    def status(self) -> dict:
        self.touch()
        return {
            "camera_id": self.camera.id,
            "faces": [asdict(f) for f in self._faces],
            "error": self._error,
        }

    def stop(self) -> None:
        self._stopped.set()
        with self._cond:
            self._cond.notify_all()

    # --- Основной цикл ---

    def run(self) -> None:
        detector = FaceDetector()
        force_refresh = False
        while not self._idle() and not self.stopped:
            try:
                self._consume(detector, force_refresh)
                force_refresh = False
            except av.error.HTTPUnauthorizedError:
                _LOGGER.info("Камера %s: 401, обновляю токен", self.camera.id)
                self._error = "Токен отклонён, обновляю"
                force_refresh = True
            except AuthError as exc:
                self._error = str(exc)
                _LOGGER.warning("Камера %s: %s", self.camera.id, exc)
                break
            except (av.error.FFmpegError, OSError) as exc:
                self._error = f"Ошибка потока: {exc}"
                _LOGGER.warning("Камера %s: %s", self.camera.id, exc)
            self._stopped.wait(_RECONNECT_DELAY)
        _LOGGER.info("Камера %s: воркер остановлен", self.camera.id)
        self.stop()

    def _consume(self, detector: FaceDetector, force_refresh: bool) -> None:
        token = self._client.access_token(force_refresh=force_refresh)
        # Заголовок передаётся и в запросы сегментов — HLS-демуксер ffmpeg копирует его
        options = {"headers": f"Authorization: Bearer {token}\r\n", "rw_timeout": "15000000"}
        _LOGGER.info("Камера %s: подключаюсь к %s", self.camera.id, self.camera.hls_url)
        with av.open(self.camera.hls_url, options=options, timeout=15) as container:
            self._error = None
            stream = container.streams.video[0]
            # Сеть читается в отдельном потоке: иначе, пока ffmpeg синхронно
            # скачивает следующий сегмент, кадры не выходят и картинка замирает.
            packets: queue.Queue = queue.Queue(maxsize=_PACKET_BUFFER)
            done = threading.Event()
            reader = threading.Thread(
                target=_read_packets, args=(container, stream, packets, done),
                name=f"camera-{self.camera.id}-reader", daemon=True,
            )
            reader.start()
            try:
                self._play(detector, packets)
            finally:
                done.set()
                _drain(packets)
                reader.join()

    def _play(self, detector: FaceDetector, packets: queue.Queue) -> None:
        """Декодирует пакеты из очереди и публикует кадры в темпе их PTS."""
        min_interval = 1 / config.DETECT_MAX_FPS
        last_detect = 0.0
        # HLS приходит сегментами по несколько секунд, и декодер выдаёт весь
        # сегмент разом. Без выравнивания по PTS клиент увидел бы только
        # последний кадр сегмента — картинка обновлялась бы раз в сегмент.
        pts_base: float | None = None
        wall_base = 0.0
        while not self.stopped and not self._idle():
            try:
                item = packets.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            for frame in item.decode():
                if frame.time is not None:
                    now = time.monotonic()
                    if pts_base is None or abs(now - (wall_base + frame.time - pts_base)) > _MAX_DRIFT:
                        # Первый кадр, разрыв в метках или сильное отставание — синхронизируемся заново
                        pts_base, wall_base = frame.time, now
                    delay = wall_base + frame.time - pts_base - now
                    if delay > 0 and self._stopped.wait(delay):
                        return
                img = frame.to_ndarray(format="bgr24")
                now = time.monotonic()
                if now - last_detect >= min_interval:
                    self._faces = detector.detect(img)
                    last_detect = now
                draw_faces(img, self._faces)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                if ok:
                    with self._cond:
                        self._jpeg = buf.tobytes()
                        self._frame_no += 1
                        self._cond.notify_all()

    def _idle(self) -> bool:
        return time.monotonic() - self._last_access > config.WORKER_IDLE_TIMEOUT


class WorkerRegistry:
    """Запускает воркер камеры по первому запросу и переиспользует его."""

    def __init__(self, client: RosdomofonClient) -> None:
        self._client = client
        self._workers: dict[int, CameraWorker] = {}
        self._lock = threading.Lock()

    def get(self, camera: Camera) -> CameraWorker:
        with self._lock:
            worker = self._workers.get(camera.id)
            if worker is None or worker.stopped:
                worker = CameraWorker(self._client, camera)
                self._workers[camera.id] = worker
                worker.start()
            worker.touch()
            return worker

    def stop_all(self) -> None:
        with self._lock:
            for worker in self._workers.values():
                worker.stop()
            self._workers.clear()
