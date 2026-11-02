"""Воркеры камер: чтение HLS с Bearer-токеном, анализ лиц, отдача JPEG-кадров."""

import logging
import queue
import threading
import time
import av
import cv2

from . import config
from .pipeline import FacePipeline
from .rosdomofon import AuthError, Camera, RosdomofonClient
from .store import Store
from .unlock import DoorUnlocker

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
    """Фоновый поток одной камеры: чтение HLS, анализ лиц, JPEG для зрителей.

    Постоянный воркер (мониторинг) работает, пока его не остановят; воркер по
    требованию останавливается сам через WORKER_IDLE_TIMEOUT без зрителей.
    """

    def __init__(
        self,
        client: RosdomofonClient,
        store: Store,
        camera: Camera,
        persistent: bool = False,
        unlocker: DoorUnlocker | None = None,
    ) -> None:
        super().__init__(name=f"camera-{camera.id}", daemon=True)
        self._client = client
        self._store = store
        self._unlocker = unlocker
        self.camera = camera
        self.persistent = persistent
        self._pipeline: FacePipeline | None = None
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_no = 0
        self._error: str | None = None
        self._streaming = False
        self._last_view = time.monotonic()
        self._stopped = threading.Event()

    # --- Для HTTP-обработчиков ---

    def touch(self) -> None:
        """Отмечает обращение зрителя (видео или текущие лица)."""
        self._last_view = time.monotonic()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def wait_frame(self, after: int, timeout: float = 10) -> tuple[int, bytes | None]:
        """Ждёт кадр новее `after`; возвращает (номер, JPEG)."""
        self.touch()
        with self._cond:
            self._cond.wait_for(lambda: self._frame_no > after or self.stopped, timeout)
            return self._frame_no, self._jpeg

    def faces(self) -> dict:
        self.touch()
        return {
            "camera_id": self.camera.id,
            "faces": self._pipeline.faces() if self._pipeline else [],
            "error": self._error,
        }

    def status(self) -> dict:
        return {
            "camera_id": self.camera.id,
            "name": self.camera.name,
            "monitored": self.persistent,
            "running": not self.stopped,
            "streaming": self._streaming,
            "error": self._error,
        }

    def stop(self) -> None:
        self._stopped.set()
        with self._cond:
            self._cond.notify_all()

    # --- Основной цикл ---

    def run(self) -> None:
        try:
            self._pipeline = FacePipeline(self.camera.id, self._store, unlocker=self._unlocker)
        except Exception as exc:  # noqa: BLE001 — например, не скачалась модель
            _LOGGER.exception("Камера %s: не удалось создать конвейер", self.camera.id)
            self._error = f"Ошибка запуска анализа: {exc}"
            self.stop()
            return
        force_refresh = False
        while not self._idle() and not self.stopped:
            try:
                self._consume(force_refresh)
                force_refresh = False
            except av.error.HTTPUnauthorizedError:
                _LOGGER.info("Камера %s: 401, обновляю токен", self.camera.id)
                self._error = "Токен отклонён, обновляю"
                force_refresh = True
            except AuthError as exc:
                self._error = f"Нужно войти заново: {exc}"
                _LOGGER.warning("Камера %s: %s", self.camera.id, exc)
                break
            except (av.error.FFmpegError, OSError) as exc:
                self._error = f"Ошибка потока: {exc}"
                _LOGGER.warning("Камера %s: %s", self.camera.id, exc)
            finally:
                self._streaming = False
                self._pipeline.flush()
            self._stopped.wait(_RECONNECT_DELAY)
        _LOGGER.info("Камера %s: воркер остановлен", self.camera.id)
        self.stop()

    def _consume(self, force_refresh: bool) -> None:
        token = self._client.access_token(force_refresh=force_refresh)
        # Заголовок передаётся и в запросы сегментов — HLS-демуксер ffmpeg копирует его
        options = {"headers": f"Authorization: Bearer {token}\r\n", "rw_timeout": "15000000"}
        _LOGGER.info("Камера %s: подключаюсь к %s", self.camera.id, self.camera.hls_url)
        with av.open(self.camera.hls_url, options=options, timeout=15) as container:
            self._error = None
            self._streaming = True
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
                self._play(packets)
            finally:
                done.set()
                _drain(packets)
                reader.join()

    def _play(self, packets: queue.Queue) -> None:
        """Декодирует пакеты из очереди и обрабатывает кадры в темпе их PTS."""
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
                self._pipeline.process(img, time.monotonic(), time.time())
                # Без зрителей рамки не рисуем и JPEG не кодируем — экономия CPU
                if self._has_viewer():
                    self._publish(img)

    def _publish(self, img) -> None:
        self._pipeline.draw(img)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        if ok:
            with self._cond:
                self._jpeg = buf.tobytes()
                self._frame_no += 1
                self._cond.notify_all()

    def _has_viewer(self) -> bool:
        return time.monotonic() - self._last_view < config.VIEWER_TIMEOUT

    def _idle(self) -> bool:
        if self.persistent:
            return False
        return time.monotonic() - self._last_view > config.WORKER_IDLE_TIMEOUT


class WorkerRegistry:
    """Воркеры камер: постоянные для мониторинга и по требованию для просмотра."""

    def __init__(self, client: RosdomofonClient, store: Store, unlocker: DoorUnlocker | None = None) -> None:
        self._client = client
        self._store = store
        self._unlocker = unlocker
        self._workers: dict[int, CameraWorker] = {}
        self._lock = threading.Lock()

    def start_monitoring(self) -> None:
        """Запускает постоянные воркеры для камер мониторинга (MONITOR_CAMERAS или все)."""
        cameras = self._client.cameras()
        wanted = config.MONITOR_CAMERAS
        if wanted is not None:
            missing = wanted - {c.id for c in cameras}
            if missing:
                _LOGGER.warning("Камеры мониторинга не найдены в аккаунте: %s", sorted(missing))
            cameras = [c for c in cameras if c.id in wanted]
        for camera in cameras:
            self._start(camera, persistent=True)
        _LOGGER.info("Мониторинг камер: %s", [c.id for c in cameras])

    def get(self, camera: Camera) -> CameraWorker:
        """Воркер для просмотра: уже работающий или новый по требованию."""
        worker = self._start(camera, persistent=False)
        worker.touch()
        return worker

    def statuses(self) -> list[dict]:
        with self._lock:
            return [w.status() for w in self._workers.values()]

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=20)

    def _start(self, camera: Camera, persistent: bool) -> CameraWorker:
        with self._lock:
            worker = self._workers.get(camera.id)
            if worker is None or worker.stopped:
                worker = CameraWorker(self._client, self._store, camera, persistent, self._unlocker)
                self._workers[camera.id] = worker
                worker.start()
            elif persistent:
                worker.persistent = True  # воркер просмотра становится постоянным
            return worker
