import pytest

from config.settings import AgentSettings
from model.factory import create_model_client
from model.nvidia_client import NVIDIAClient
from model.ollama_client import OllamaClient


def test_model_provider_defaults_to_nvidia(monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    settings = AgentSettings()

    client = create_model_client(settings)

    assert settings.model_provider == "nvidia"
    assert settings.nvidia_model == "meta/llama-3.3-70b-instruct"
    assert isinstance(client, NVIDIAClient)
    assert client.base_url == settings.nvidia_endpoint
    assert client.model == settings.nvidia_model


def test_ollama_can_be_selected_explicitly(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setattr(OllamaClient, "ensure_model_available", lambda self: [])

    settings = AgentSettings()

    client = create_model_client(settings)

    assert isinstance(client, OllamaClient)


def test_unknown_model_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="MODEL_PROVIDER"):
        create_model_client(AgentSettings())
