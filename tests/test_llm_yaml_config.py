from service.llm.llm_client import LLMService


def test_llm_service_prefers_yaml_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setenv("ANYROUTER_API_KEY", "env-anyrouter-key")
    monkeypatch.setenv("ANYROUTER_BASE_URL", "https://env.example.com/v1")

    service = LLMService(
        {
            "llm": {
                "model": "yaml-model",
                "api_key": "yaml-key",
                "base_url": "https://yaml.example.com/v1",
                "api_key_env": "ANYROUTER_API_KEY",
                "base_url_env": "ANYROUTER_BASE_URL",
                "enable_real_generation": True,
                "timeout_seconds": 12,
            }
        }
    )

    assert service.model == "yaml-model"
    assert service.api_key == "yaml-key"
    assert service.base_url == "https://yaml.example.com/v1"
    assert service.timeout_seconds == 12


def test_llm_service_reads_provider_model_and_endpoint(monkeypatch):
    monkeypatch.setenv("YAML_PROVIDER_KEY", "provider-key")

    service = LLMService(
        {
            "llm": {
                "current_model": "anyrouter-gpt-5.3-codex",
                "provider": "anyrouter",
                "enable_real_generation": True,
                "providers": {
                    "anyrouter": {
                        "base_url": "https://provider.example.com/v1",
                        "api_key_env": "YAML_PROVIDER_KEY",
                        "use_responses_api": True,
                        "models": {
                            "gpt-5.3-codex": {"model": "gpt-5.3-codex"},
                        },
                    }
                },
            }
        }
    )

    assert service.provider_name == "anyrouter"
    assert service.model == "gpt-5.3-codex"
    assert service.api_key == "provider-key"
    assert service.base_url == "https://provider.example.com/v1"
    assert service.use_responses_api is True
    assert service.trace_metadata()["base_url_set"] is True


def test_llm_service_accepts_literal_key_in_api_key_env_for_legacy_yaml():
    service = LLMService(
        {
            "llm": {
                "api_key_env": "***REMOVED_API_KEY***",
                "enable_real_generation": True,
            }
        }
    )

    assert service.api_key == "***REMOVED_API_KEY***"


def test_llm_service_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("TRUSTED_QA_ENABLE_REAL_LLM", "0")

    service = LLMService({"llm": {"api_key": "yaml-key", "enable_real_generation": True}})

    assert service.enabled is False
    assert service.is_available is False
