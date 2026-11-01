"""Рамки и подписи поверх кадра. Подписи — Pillow: в шрифтах OpenCV нет кириллицы."""

import threading
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config
from .detector import ensure_model

# Цвета в BGR (OpenCV) по статусу появления
COLORS = {
    "known": (60, 190, 60),
    "unknown": (0, 140, 255),
    "pending": (160, 160, 160),
}
_FONT_SIZE = 16
_PADDING = 3

_font_lock = threading.Lock()
_font: ImageFont.FreeTypeFont | None = None


def _get_font() -> ImageFont.FreeTypeFont:
    global _font
    with _font_lock:
        if _font is None:
            ensure_model(config.FONT_FILE, config.FONT_URL, config.FONT_SHA256)
            _font = ImageFont.truetype(str(config.FONT_FILE), _FONT_SIZE)
        return _font


@lru_cache(maxsize=256)
def _label_image(text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Плашка с текстом (BGR): белый текст на фоне цвета рамки. Кэшируется."""
    font = _get_font()
    left, top, right, bottom = font.getbbox(text)
    width, height = right - left + 2 * _PADDING, bottom - top + 2 * _PADDING
    image = Image.new("RGB", (width, height), color[::-1])
    ImageDraw.Draw(image).text((_PADDING - left, _PADDING - top), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_box(frame: np.ndarray, box: tuple[int, int, int, int], status: str, text: str) -> None:
    """Рисует рамку цвета статуса и подпись над ней (или внутри, если сверху нет места)."""
    color = COLORS[status]
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = _label_image(text, color)
    lh, lw = label.shape[:2]
    fh, fw = frame.shape[:2]
    ly = y - lh if y - lh >= 0 else max(y, 0)
    lx = min(max(x, 0), max(fw - lw, 0))
    lh, lw = min(lh, fh - ly), min(lw, fw - lx)
    if lh > 0 and lw > 0:
        frame[ly:ly + lh, lx:lx + lw] = label[:lh, :lw]
