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