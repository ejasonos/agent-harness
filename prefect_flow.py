from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger

from agent.loop import AgentLoop, AgentLoopConfig
from config.settings import load_settings
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry
from model.nvidia_client import NVIDIAClient
from model.ollama_client import OllamaClient
from workspace.manager import WorkspaceManager


@flow(name="local-agent-task")
def run_agent(task: str, workspace_path: str | None = None) -> dict[str, Any]:
    """Run one agent task using the deployment's configured services."""
    logger = get_run_logger()
    workspace_root = Path(
        workspace_path or os.getenv("AGENT_WORKSPACE") or Path.cwd()
    ).expanduser().resolve()
    settings = load_settings(workspace_root)

    provider = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
    if provider == "nvidia":
        model = NVIDIAClient(
            model=settings.nvidia_model,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_endpoint,
            timeout=600,
        )
    elif provider == "ollama":
        model = OllamaClient(
            model=settings.model,
            base_url=settings.ollama_host,
            timeout=600,
        )
        model.ensure_model_available()
    else:
        raise ValueError("MODEL_PROVIDER must be 'nvidia' or 'ollama'.")

    workspace = WorkspaceManager(
        settings.require_workspace(),
        allow_delete=settings.allow_delete,
        allow_write=True,
        allow_git_write=settings.allow_git_write,
        allow_shell=settings.allow_shell,
        allow_network=settings.allow_network,
    )
    mcp_client = MCPClient(settings)
    agent = AgentLoop(
        model=model,
        mcp_client=mcp_client,
        registry=ToolRegistry(mcp_client),
        workspace=workspace,
        config=AgentLoopConfig(
            max_iterations=settings.max_iterations,
            max_tool_calls=settings.max_tool_calls,
            max_context_messages=settings.max_context_messages,
            max_context_tokens=settings.max_context_tokens,
            model_input_cost_per_million_tokens=(
                settings.model_input_cost_per_million_tokens
            ),
            model_output_cost_per_million_tokens=(
                settings.model_output_cost_per_million_tokens
            ),
            enable_subagents=settings.enable_subagents,
            max_subagents_per_run=settings.max_subagents_per_run,
            max_subagent_iterations=settings.max_subagent_iterations,
            max_subagent_tool_calls=settings.max_subagent_tool_calls,
            max_subagent_task_chars=settings.max_subagent_task_chars,
        ),
    )

    logger.info(
        "Starting agent task with provider=%s workspace=%s",
        provider,
        workspace_root,
    )
    try:
        state = agent.run(task)
    finally:
        mcp_client.http.close()
    result = {
        "completed": state.completed,
        "iterations": state.iteration,
        "tool_execution_count": len(state.tool_executions),
        "subagent_tool_execution_count": state.metadata.get(
            "subagent_tool_execution_count",
            0,
        ),
        "subagent_count": len(state.metadata.get("delegated_tasks", [])),
        "model_call_count": state.model_call_count,
        "model_usage": state.model_usage,
        "model_duration_ms": state.model_duration_ms,
        "estimated_model_cost_usd": state.estimated_model_cost_usd,
        "run_duration_ms": state.run_duration_ms,
        "tool_executions": [
            {
                "tool_name": execution.tool_name,
                "success": execution.success,
                "error": execution.error,
                "duration_ms": execution.duration_ms,
            }
            for execution in state.tool_executions
        ],
        "final_answer": state.final_answer,
        "error": state.error,
    }
    logger.info(
        "Agent task finished: completed=%s iterations=%s tool_executions=%s",
        state.completed,
        state.iteration,
        len(state.tool_executions),
    )
    if state.error:
        raise RuntimeError(state.error)
    return result