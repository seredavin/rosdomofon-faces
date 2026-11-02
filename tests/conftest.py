import pytest

from app import main
from app.session import Auth


@pytest.fixture(autouse=True)
def _ui_auth_disabled(tmp_path, monkeypatch):
    """Тесты API не зависят от ROSDOMOFON_UI_PASSWORD в окружении разработчика."""
    monkeypatch.setattr(main, "auth", Auth("", tmp_path / "session_secret"))
