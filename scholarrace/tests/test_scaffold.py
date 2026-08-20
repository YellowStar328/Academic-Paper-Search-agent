"""Smoke test for project scaffold."""

from app.config import Settings, get_settings
from app.main import app


def test_settings_defaults():
    settings = Settings(app_env="development")
    assert settings.app_env == "development"
    assert settings.embedding_top_k == 100
    assert settings.w_relevance == 0.35
    assert settings.thompson_exploration_floor == 0.10


def test_test_env_database_url():
    settings = Settings(app_env="test")
    assert settings.is_test is True
    assert "sqlite" in settings.effective_database_url


def test_get_settings_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_health_endpoint():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
