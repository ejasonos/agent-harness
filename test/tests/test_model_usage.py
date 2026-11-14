from agent.state import AgentState
from config.settings import AgentSettings
from model.parser import ModelParser


def test_openai_compatible_usage_is_normalized():
    response = ModelParser.parse(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 7,
                "total_tokens": 32,
            },
        }
    )

    assert response.usage == {
        "prompt_tokens": 25,
        "completion_tokens": 7,
        "total_tokens": 32,
    }


def test_cost_estimate_requires_usage_and_configured_rates():
    state = AgentState(task="test")
    state.record_model_call(
        {"prompt_tokens": 100, "completion_tokens": 20},
        input_cost_per_million_tokens=1.0,
        output_cost_per_million_tokens=2.0,
    )
    state.record_model_call({})

    assert state.model_call_count == 2
    assert state.model_usage == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 0,
    }
    assert state.estimated_model_cost_usd == 0.00014


def test_context_and_cost_settings_are_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("MAX_CONTEXT_TOKENS", "12000")
    monkeypatch.setenv("MODEL_INPUT_COST_PER_MILLION_TOKENS", "0.5")
    monkeypatch.setenv("MODEL_OUTPUT_COST_PER_MILLION_TOKENS", "1.25")

    settings = AgentSettings()

    assert settings.max_context_tokens == 12000
    assert settings.model_input_cost_per_million_tokens == 0.5
    assert settings.model_output_cost_per_million_tokens == 1.25


def test_subagent_limits_are_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("ENABLE_SUBAGENTS", "false")
    monkeypatch.setenv("MAX_SUBAGENTS_PER_RUN", "3")
    monkeypatch.setenv("MAX_SUBAGENT_ITERATIONS", "6")
    monkeypatch.setenv("MAX_SUBAGENT_TOOL_CALLS", "9")

    settings = AgentSettings()

    assert settings.enable_subagents is False
    assert settings.max_subagents_per_run == 3
    assert settings.max_subagent_iterations == 6
    assert settings.max_subagent_tool_calls == 9