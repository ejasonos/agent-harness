from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent.context import ContextManager
from agent.loop import AgentLoop, AgentLoopConfig
from config.settings import load_settings
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry
from model.nvidia_client import NVIDIAClient
from model.ollama_client import OllamaClient
from workspace.manager import WorkspaceManager


load_dotenv()

st.set_page_config(
    page_title="Local Agent",
    page_icon="LA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17251f;
        --muted: #68766f;
        --paper: #f4f7f4;
        --line: #dce5de;
        --green: #216b50;
        --mint: #d8f1e3;
        --amber: #bd6c22;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #eaf0eb;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem;
    }

    .agent-mark {
        color: var(--green);
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .agent-title {
        color: var(--ink);
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0.4rem 0 0.25rem;
    }

    .agent-subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.72);
    }

    div[data-testid="stChatInput"] textarea {
        border-color: #c6d6cb;
        background: white;
    }

    .stButton > button[kind="primary"] {
        background: var(--green);
        border-color: var(--green);
    }

    code, pre, .stCode {
        font-family: 'DM Mono', monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _build_agent(
    workspace_path: str,
    mcp_url: str,
    model_provider: str,
    model_name: str,
    ollama_url: str,
    settings: Any,
) -> AgentLoop:
    workspace_root = Path(workspace_path).expanduser().resolve()
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise ValueError(f"Workspace directory does not exist: {workspace_root}")

    settings.set_workspace(workspace_root)
    settings.mcp_url = mcp_url.strip()

    if model_provider == "NVIDIA API":
        model = NVIDIAClient(
            model=model_name,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_endpoint,
            timeout=600,
        )
    else:
        model = OllamaClient(
            model=model_name,
            base_url=ollama_url,
            timeout=600,
        )

    mcp_client = MCPClient(settings)
    registry = ToolRegistry(mcp_client)
    workspace = WorkspaceManager(
        workspace_root,
        allow_delete=settings.allow_delete,
        allow_write=True,
        allow_git_write=settings.allow_git_write,
        allow_shell=settings.allow_shell,
        allow_network=settings.allow_network,
    )

    return AgentLoop(
        model=model,
        mcp_client=mcp_client,
        registry=registry,
        workspace=workspace,
        context=ContextManager(
            max_messages=settings.max_context_messages
        ),
        config=AgentLoopConfig(
            max_iterations=settings.max_iterations,
            max_tool_calls=settings.max_tool_calls,
        ),
    )


def _render_run(record: dict[str, Any]) -> None:
    if record.get("error"):
        st.error(record["error"])
    elif record.get("answer"):
        st.markdown(record["answer"])
    else:
        st.info("The run ended without a final answer.")

    executions = record.get("executions", [])
    if executions:
        with st.expander(
            f"Tool activity · {len(executions)} calls",
            expanded=False,
        ):
            for index, execution in enumerate(executions, start=1):
                status = "SUCCESS" if execution["success"] else "FAILED"
                label = f"{index:02d}  {status}  {execution['tool_name']}"
                with st.expander(label, expanded=False):
                    if execution.get("reason_for_tool"):
                        st.text(
                            f"Reason: {execution['reason_for_tool']}"
                        )
                    st.json(execution["arguments"])
                    if execution.get("error"):
                        st.error(execution["error"])
                    if execution.get("result") is not None:
                        result = execution["result"]
                        if isinstance(result, str):
                            st.code(result)
                        else:
                            st.json(result)

    if record.get("iterations") is not None:
        first, second = st.columns(2)
        first.metric("Iterations", record["iterations"])
        second.metric("Tool calls", len(executions))


def _run_task(
    task: str,
    workspace_path: str,
    mcp_url: str,
    model_provider: str,
    model_name: str,
    ollama_url: str,
) -> dict[str, Any]:
    settings = load_settings()
    agent = _build_agent(
        workspace_path,
        mcp_url,
        model_provider,
        model_name,
        ollama_url,
        settings,
    )
    state = agent.run(task)

    return {
        "answer": state.final_answer,
        "error": state.error,
        "completed": state.completed,
        "iterations": state.iteration,
        "executions": [
            {
                "tool_name": execution.tool_name,
                "arguments": execution.arguments,
                "reason_for_tool": execution.reason_for_tool,
                "success": execution.success,
                "result": execution.result,
                "error": execution.error,
                "duration_ms": execution.duration_ms,
            }
            for execution in state.tool_executions
        ],
    }


with st.sidebar:
    st.markdown('<div class="agent-mark">Agent console / 01</div>', unsafe_allow_html=True)
    st.subheader("Session")

    workspace_path = st.text_input(
        "Agent workspace",
        value=os.getenv("AGENT_WORKSPACE", str(Path.cwd())),
        help="The MCP server must use the same workspace path for file tools.",
    )
    mcp_url = st.text_input(
        "MCP endpoint",
        value=os.getenv("MCP_URL", "http://127.0.0.1:8082/mcp"),
    )
    model_provider = st.selectbox(
        "Model provider",
        ["Ollama", "NVIDIA API"],
        index=1 if os.getenv("NVIDIA_API_KEY") else 0,
    )

    if model_provider == "NVIDIA API":
        model_name = st.text_input(
            "NVIDIA model",
            value=os.getenv(
                "NVIDIA_MODEL",
                "meta/llama-3.2-11b-vision-instruct",
            ),
            key="nvidia_model_name",
        )
    else:
        model_name = st.text_input(
            "Ollama model",
            value=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
            key="ollama_model_name",
        )
    ollama_url = st.text_input(
        "Ollama URL",
        value=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        disabled=model_provider != "Ollama",
    )

    st.divider()
    if "fastmcp.app" in mcp_url:
        st.warning(
            "Horizon OAuth is not implemented in this agent's MCP client. "
            "A protected endpoint will not connect yet."
        )
    else:
        st.caption("Secrets are read from environment variables or .env.")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state["agent_history"] = []
        st.rerun()


if "agent_history" not in st.session_state:
    st.session_state["agent_history"] = []

st.markdown('<div class="agent-mark">Local tools · Model-directed execution</div>', unsafe_allow_html=True)
st.markdown('<div class="agent-title">Local Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="agent-subtitle">Task runs, evidence, and tool outcomes in one place.</div>',
    unsafe_allow_html=True,
)

for entry in st.session_state["agent_history"]:
    with st.chat_message(entry["role"]):
        if entry["role"] == "user":
            st.markdown(entry["content"])
        else:
            _render_run(entry["content"])

task = st.chat_input("Give the agent a task")

if task:
    with st.chat_message("user"):
        st.markdown(task)
    st.session_state["agent_history"].append(
        {"role": "user", "content": task}
    )

    with st.chat_message("assistant"):
        with st.spinner("Agent is working"):
            try:
                record = _run_task(
                    task,
                    workspace_path,
                    mcp_url,
                    model_provider,
                    model_name,
                    ollama_url,
                )
            except Exception as exc:
                record = {
                    "answer": None,
                    "error": str(exc),
                    "completed": True,
                    "iterations": None,
                    "executions": [],
                }
        _render_run(record)

    st.session_state["agent_history"].append(
        {"role": "assistant", "content": record}
    )