"""Запуск: python -m app [--host 127.0.0.1] [--port 8000]."""

import argparse
import logging

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Росдомофон: видео с поиском лиц")
    # По умолчанию только localhost: у сервиса пока нет своей авторизации,
    # а через него доступны камеры и токены аккаунта Росдомофона.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run("app.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
