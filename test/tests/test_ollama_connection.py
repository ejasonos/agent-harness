from unittest.mock import Mock

import pytest
import requests

from model.messages import Message, ToolCall
from model.ollama_client import OllamaClient


def test_preflight_checks_local_tags_without_generation(monkeypatch):
    client = OllamaClient(model="gemma4:e4b")
    list_models = Mock(return_value=["gemma4:e4b", "gemma4:12b"])
    post = Mock(side_effect=AssertionError("generation must not be called"))
    monkeypatch.setattr(client, "list_models", list_models)
    monkeypatch.setattr("model.ollama_client.requests.post", post)

    assert client.ensure_model_available() == ["gemma4:e4b", "gemma4:12b"]
    list_models.assert_called_once_with()
    post.assert_not_called()


def test_preflight_explains_when_model_is_not_installed(monkeypatch):
    client = OllamaClient(model="missing-model:latest")
    monkeypatch.setattr(client, "list_models", lambda: ["gemma4:e4b"])

    with pytest.raises(RuntimeError, match="ollama pull missing-model:latest"):
        client.ensure_model_available()


def test_preflight_explains_when_local_server_is_unreachable(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(
        client,
        "list_models",
        Mock(side_effect=requests.ConnectionError("connection refused")),
    )

    with pytest.raises(ConnectionError, match="127.0.0.1:11434"):
        client.ensure_model_available()


def test_cloud_tagged_model_is_rejected_before_inference():
    with pytest.raises(ValueError, match="not local"):
        OllamaClient(model="gpt-oss:120b-cloud")


def test_cloud_model_requires_explicit_opt_in_and_preflight_is_generation_free(
    monkeypatch,
):
    client = OllamaClient(
        model="beam:cloud",
        allow_cloud_models=True,
    )
    post = Mock(side_effect=AssertionError("generation must not be called"))
    monkeypatch.setattr(client, "list_models", lambda: ["beam:cloud"])
    monkeypatch.setattr("model.ollama_client.requests.post", post)

    assert client.ensure_model_available() == ["beam:cloud"]
    post.assert_not_called()


def test_ollama_serializes_tool_round_trip_in_native_format():
    messages = [
        Message(role="user", content="Inspect README.md"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    "read_file",
                    {"path": "README.md"},
                    call_id="openai-style-id",
                    reason_for_tool="Read the requested file.",
                )
            ],
        ),
        Message(
            role="tool",
            name="read_file",
            tool_call_id="openai-style-id",
            content="Project documentation",
        ),
    ]

    normalized = OllamaClient._normalize_messages(
        OllamaClient(),
        messages,
    )

    assert normalized == [
        {"role": "user", "content": "Inspect README.md"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": {
                            "path": "README.md",
                            "reason_for_tool": "Read the requested file.",
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "Project documentation"},
    ]