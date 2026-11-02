"""Тесты клиента Росдомофона: двери и открытие (HTTP подменяется MockTransport)."""

import json
import time

import httpx
import pytest

from app import config
from app.rosdomofon import DoorError, RosdomofonClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TOKEN_FILE", tmp_path / "token.json")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    c = RosdomofonClient()
    c._tok = {"access_token": "old", "refresh_token": "r", "expires_in": 3600, "timestamp": time.time()}
    return c


def _mock(client, handler):
    client._http = httpx.Client(transport=httpx.MockTransport(handler))


def test_doors_parses_keys(client):
    def handler(request):
        assert request.url == config.KEYS_LIST_URL
        return httpx.Response(200, json=[
            {"adapterId": "aabbccddeeff", "relay": 1, "type": 1},
            {"adapterId": "aabbccddeeff", "relay": 2, "type": 4},
            {"type": 1},  # без adapterId — пропускается
        ])
    _mock(client, handler)
    doors = client.doors()
    assert [(d.id, d.name) for d in doors] == [("aabbccddeeff:1", "Дверь подъезда"), ("aabbccddeeff:2", "Калитка")]


def test_open_door_sends_relay(client):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200)
    _mock(client, handler)
    client.open_door("aabbccddeeff", 1)
    assert seen == {
        "url": config.OPEN_DOOR_URL.format(adapter_id="aabbccddeeff"),
        "body": {"rele": 1},
        "auth": "Bearer old",
    }


def test_open_door_refreshes_token_on_401(client):
    calls = []

    def handler(request):
        if request.url == config.TOKEN_REQUEST_URL:
            return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})
        calls.append(request.headers["Authorization"])
        return httpx.Response(401 if len(calls) == 1 else 200)
    _mock(client, handler)
    client.open_door("a", 1)
    assert calls == ["Bearer old", "Bearer new"]


def test_open_door_errors(client):
    _mock(client, lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(DoorError, match="500"):
        client.open_door("a", 1)

    def fail(request):
        raise httpx.ConnectError("нет сети")
    _mock(client, fail)
    with pytest.raises(DoorError, match="недоступен"):
        client.open_door("a", 1)
