from __future__ import annotations

from core.config_loader import _map_to_env
from utils.config_loader import get_app_config


def test_app_config_declares_mineru_and_embedding_api_key_envs():
    config = get_app_config(reload=True)

    assert config["embedding"]["api_key_env"] == "QWEN_API_KEY"
    assert "api_key" in config["embedding"]
    assert "mineru" not in config
    assert isinstance(config["api_keys"]["mineru_api_key"], str)
    assert config["api_keys"]["mineru_api_key_env"] == "MinerU_API_KEY"
    assert config["llm"]["providers"]["anyrouter"]["api_key_env"] == "ANYROUTER_API_KEY"


def test_runtime_env_maps_mineru_and_embedding_key_config(monkeypatch):
    monkeypatch.delenv("CUSTOM_EMBEDDING_KEY", raising=False)
    monkeypatch.delenv("CUSTOM_MINERU_KEY", raising=False)

    env = _map_to_env(
        {
            "embedding": {
                "provider": "qwen",
                "model": "text-embedding-v4",
                "dimension": 1024,
                "api_key": "embedding-test-key",
                "api_key_env": "CUSTOM_EMBEDDING_KEY",
                "base_url": "https://embedding.example.com/v1",
            },
            "api_keys": {
                "mineru_api_key": "mineru-test-key",
                "mineru_api_key_env": "CUSTOM_MINERU_KEY",
            },
        }
    )

    assert env["EMBEDDING_API_KEY"] == "embedding-test-key"
    assert env["QWEN_API_KEY"] == "embedding-test-key"
    assert env["CUSTOM_EMBEDDING_KEY"] == "embedding-test-key"
    assert env["EMBEDDING_API_KEY_ENV"] == "CUSTOM_EMBEDDING_KEY"
    assert env["EMBEDDING_BASE_URL"] == "https://embedding.example.com/v1"
    assert env["EMBEDDING_MODEL"] == "text-embedding-v4"

    assert env["MinerU_API_KEY"] == "mineru-test-key"
    assert env["MINERU_API_KEY"] == "mineru-test-key"
    assert env["CUSTOM_MINERU_KEY"] == "mineru-test-key"
    assert env["MINERU_API_KEY_ENV"] == "CUSTOM_MINERU_KEY"
