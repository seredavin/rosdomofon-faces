"""Запуск: python -m app [--host 127.0.0.1] [--port 8000]."""

import argparse
import logging

import uvicorn

from . import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Росдомофон: видео с поиском лиц")
    # По умолчанию только localhost: без ROSDOMOFON_UI_PASSWORD авторизации нет,
    # а через сервис доступны камеры, база лиц и открытие двери.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # За reverse proxy адрес клиента (лимит попыток входа) и схема (флаг Secure
    # у cookie) берутся из X-Forwarded-*, но только от доверенных адресов.
    uvicorn.run(
        "app.main:app", host=args.host, port=args.port,
        proxy_headers=True, forwarded_allow_ips=config.FORWARDED_ALLOW_IPS,
    )


if __name__ == "__main__":
    main()
