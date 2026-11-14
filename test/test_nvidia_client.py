from unittest.mock import Mock

from model.messages import Message, ToolCall
from model.nvidia_client import NVIDIAClient


def test_nvidia_normalizes_completed_tool_batch_as_single_call_turns():
    calls = [
        ToolCall(
            "write_file",
            {"path": f"poem{index}.txt"},
            f"call-{index}",
        )
        for index in range(1, 4)
    ]
    messages = [
        Message(
            role="assistant",
            tool_calls=calls,
        ),
        *[
            Message(
                role="tool",
                name="write_file",
                tool_call_id=f"call-{index}",
                content=f"created poem{index}.txt",
            )
            for index in range(1, 4)
        ],
    ]

    normalized = NVIDIAClient(
        model="test-model",
        api_key="test-key",
    )._normalize_messages(messages)

    assert [
        message["role"]
        for message in normalized
    ] == ["assistant", "tool"] * 3
    assert [
        message["tool_calls"][0]["id"]
        for message in normalized
        if message["role"] == "assistant"
    ] == ["call-1", "call-2", "call-3"]
    assert [
        message["tool_call_id"]
        for message in normalized
        if message["role"] == "tool"
    ] == ["call-1", "call-2", "call-3"]


def test_nvidia_chat_disables_parallel_tool_calls(monkeypatch):
    monkeypatch.setenv("LOG_PAYLOADS", "true")
    events = []
    monkeypatch.setattr(
        "model.nvidia_client.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    response = Mock(ok=True)
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "done",
                },
                "finish_reason": "stop",
            }
        ]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("model.nvidia_client.requests.post", post)
    client = NVIDIAClient(
        model="test-model",
        api_key="test-key",
    )

    client.chat(
        messages=[Message(role="user", content="create files")],
        tools=[{"type": "function", "function": {"name": "write_file"}}],
    )

    payload = post.call_args.kwargs["json"]
    assert payload["parallel_tool_calls"] is False
    assert [event for event, _fields in events] == [
        "model.request.sent",
        "model.request.payload",
        "model.response.received",
        "model.response.payload",
        "model.response.parsed",
        "model.response.details",
    ]


def test_nvidia_usage_is_normalized_and_payload_logs_are_opt_in(monkeypatch):
    monkeypatch.delenv("LOG_PAYLOADS", raising=False)
    events = []
    monkeypatch.setattr(
        "model.nvidia_client.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    response = Mock(ok=True, status_code=200, text="response")
    response.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
        },
    }
    monkeypatch.setattr("model.nvidia_client.requests.post", Mock(return_value=response))
    client = NVIDIAClient(model="test-model", api_key="test-key")

    result = client.chat([Message(role="user", content="private task")])

    assert result.usage["prompt_tokens"] == 12
    assert result.usage["completion_tokens"] == 4
    assert result.usage["total_tokens"] == 16
    assert "request_duration_ms" in result.usage
    assert not any("payload" in event or event.endswith("details") for event, _ in events)