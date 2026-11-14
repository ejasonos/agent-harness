# Local Agent

Local Agent is a model-directed AI agent. It connects a model to an HTTP MCP server that exposes workspace, Git, shell, browser, conversion, screenshot, and testing tools. Use the Python `AgentLoop` API to submit tasks and inspect tool activity.

## Quick Start

Run commands from the project directory. The MCP server and the Python program that runs the agent are separate processes, so keep two terminals open.

### 1. Install dependencies

Python 3.10 or later is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `\.venv\Scripts\python.exe` in place of `python` for the remaining commands.

### 2. Configure a model

The default provider is NVIDIA's hosted inference API. Set `NVIDIA_API_KEY` in `.env`; the default endpoint is `https://integrate.api.nvidia.com/v1` and the default model is `meta/llama-3.3-70b-instruct`. Ollama remains available by setting `MODEL_PROVIDER=ollama` and installing the configured local model:

```powershell
ollama pull gemma4:e4b
ollama list
```

The Ollama desktop app normally starts its local service. If the service is not running, start it in a separate terminal with `ollama serve`. Confirm the model responds with `ollama run gemma4:e4b` and exit the model prompt with `/bye`.

The development runner, `test_agent_loop.py`, uses the provider selected by `MODEL_PROVIDER` (NVIDIA by default). NVIDIA inference may consume provider quota or credits.

### 3. Start the MCP server

In the first terminal, from the project directory:

```powershell
python fastmcp_tools_http.py
```

The server listens on `http://127.0.0.1:8082/mcp` by default. Leave this process running while using the app.

### 4. Run an agent task

In a second terminal, from the project directory, create or run a Python program that constructs the agent and calls `agent.run(task)`. For example:

```python
from dotenv import load_dotenv

from agent.loop import AgentLoop, AgentLoopConfig
from config.settings import load_settings
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry
from model.factory import create_model_client
from workspace.manager import WorkspaceManager

load_dotenv()

workspace_path = r"D:\work\my-project"
settings = load_settings(workspace_path)
mcp_client = MCPClient(settings)
workspace = WorkspaceManager(
	settings.require_workspace(),
	allow_delete=settings.allow_delete,
	allow_write=True,
	allow_git_write=settings.allow_git_write,
	allow_shell=settings.allow_shell,
	allow_network=settings.allow_network,
)
model = create_model_client(settings, timeout=600)
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

result = agent.run("Explain the project structure")
print(result.final_answer)
if result.error:
	print(result.error)
```

Replace the example task and workspace path with your own. `result.tool_executions` contains each tool's name, reason, arguments, result, and status. Set `MODEL_PROVIDER=ollama` to use Ollama instead of the default NVIDIA provider.

The MCP server and agent program must use the same workspace directory. The server uses `AGENT_WORKSPACE` or its current working directory; `load_settings(workspace_path)` configures the agent's local workspace.

### How Model Providers Are Wired In

The agent runner uses `MODEL_PROVIDER` to select the inference client. Both supported providers implement the shared `chat()` interface; the agent loop and MCP server do not need provider-specific changes:

- `config/defaults.py` sets NVIDIA as the default provider and `gemma4:e4b` as the Ollama model.
- `config/settings.py` reads `MODEL_PROVIDER` and the selected provider's endpoint, model, and credentials from the environment.
- `model/ollama_client.py` sends chat requests to Ollama's `/api/chat` endpoint and exposes the available-model health/list checks. It uses the `requests` dependency already in `requirements.txt`; no Ollama Python package is needed.
- `model/factory.py` creates the configured NVIDIA or Ollama client; `test_agent_loop.py` and the Prefect flow both use it.
- `agent/planner.py` calls the provider through the shared `chat()` interface. It can stay unchanged when switching between the Ollama and NVIDIA clients.
- `fastmcp_tools_http.py` is a separate tools server. It does not call the model and does not need Ollama configuration.

Set these values in the project `.env` file to select a provider and configure its endpoint/model:

```dotenv
MODEL_PROVIDER=nvidia
NVIDIA_API_KEY=your-key
NVIDIA_ENDPOINT=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.3-70b-instruct

# Optional local Ollama configuration (set MODEL_PROVIDER=ollama to use it)
OLLAMA_MODEL=gemma4:e4b
OLLAMA_HOST=http://127.0.0.1:11434
```

For Ollama, `OLLAMA_HOST` is the server's base URL, not the `/api/chat` endpoint, and `OLLAMA_MODEL` must match an installed model that supports tool calling. After changing `.env`, restart the agent process. Web search is a separate tool and still requires network access and `TAVILY_API_KEY`.

The Ollama provider calls `ensure_model_available()` before starting the agent. It queries only `/api/tags`, verifies the configured model is installed, and fails with a local setup hint if it is not. This preflight makes no generation request. Cloud-tagged Ollama model names are rejected to avoid silently using remote inference.

Inference internals such as KV caching, paged attention, continuous batching, and speculative decoding stay with the configured serving engine (Ollama or NVIDIA). This Python agent records provider usage and latency but does not reimplement model serving. Fine-tuning or vector retrieval should be added only when offline trajectory evaluations identify a concrete quality gap or a corpus that workspace/web search cannot cover.

Subagents are bounded investigators. They share the configured model and MCP connection but receive only read-only file, project, and Git inspection tools. A child has its own evaluator, cannot delegate, and cannot modify files or run shell commands. The parent retains authority over mutations, verification, and final completion. Delegations run serially; failed or capped children are surfaced as unresolved parent tool failures.

### Use a different workspace

In each PowerShell terminal, set `AGENT_WORKSPACE` to the existing project directory before starting the process:

```powershell
$env:AGENT_WORKSPACE = "D:\work\my-project"
```

Then start `python fastmcp_tools_http.py` in one terminal and your Python agent program in the other. PowerShell environment variables are scoped to their terminal, so set the value in both terminals. Alternatively, put `AGENT_WORKSPACE=D:\work\my-project` in the project `.env` file; the MCP server reads it at startup. Restart the server after changing the workspace.

## Configuration

The MCP server and your agent program can load a `.env` file from the project directory. Environment variables override the defaults. For example:

```dotenv
AGENT_WORKSPACE=C:\path\to\your\project
MCP_URL=http://127.0.0.1:8082/mcp
MODEL_PROVIDER=nvidia
NVIDIA_API_KEY=your-key
NVIDIA_ENDPOINT=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.3-70b-instruct

# Optional Ollama configuration
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=gemma4:e4b

# Required for web_search
TAVILY_API_KEY=your-tavily-key

# Structured agent/model/MCP trace output
LOG_LEVEL=DEBUG
LOG_FILE=logs/agent.jsonl
LOG_PAYLOADS=false

# Context budget and optional model pricing, USD per million tokens
MAX_CONTEXT_MESSAGES=50
MAX_CONTEXT_TOKENS=16000
ENABLE_SUBAGENTS=true
MAX_SUBAGENTS_PER_RUN=2
MAX_SUBAGENT_ITERATIONS=8
MAX_SUBAGENT_TOOL_CALLS=12
MAX_SUBAGENT_TASK_CHARS=2000
# MODEL_INPUT_COST_PER_MILLION_TOKENS=...
# MODEL_OUTPUT_COST_PER_MILLION_TOKENS=...

# Tool permissions
ALLOW_DELETE=true
ALLOW_GIT_WRITE=false
ALLOW_SHELL=true
ALLOW_NETWORK=true
```

Available runtime settings include:

| Setting | Default | Purpose |
| --- | --- | --- |
| `AGENT_WORKSPACE` | Current directory | Root used by the MCP server. Set the UI to the same path. |
| `MCP_URL` | `http://127.0.0.1:8082/mcp` | MCP endpoint used by the agent. |
| `MODEL_PROVIDER` | `nvidia` | Uses NVIDIA by default; set `ollama` to select local Ollama. |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API URL. |
| `OLLAMA_MODEL` | `gemma4:e4b` | Default local model. |
| `NVIDIA_API_KEY` | Unset | Required when using NVIDIA API. |
| `NVIDIA_ENDPOINT` | `https://integrate.api.nvidia.com/v1` | NVIDIA-compatible API endpoint. |
| `NVIDIA_MODEL` | `meta/llama-3.3-70b-instruct` | Default NVIDIA model. |
| `TAVILY_API_KEY` | Unset | Required by the Tavily-backed `web_search` tool. |
| `ALLOW_DELETE` | `true` | Enables delete operations. Set to `false` to disable them. |
| `ALLOW_GIT_WRITE` | `false` | Enables Git mutations. |
| `ALLOW_SHELL` | `true` | Enables shell and process tools. |
| `ALLOW_NETWORK` | `true` | Enables network-dependent tools. |
| `MAX_ITERATIONS` | `30` | Maximum agent planning iterations. |
| `MAX_TOOL_CALLS` | `100` | Maximum tool calls per run. |
| `MAX_CONTEXT_MESSAGES` | `50` | Maximum retained context messages, subject to required system/task messages. |
| `MAX_CONTEXT_TOKENS` | `16000` | Estimated context token budget. A provider tokenizer can be supplied through `ContextManager`. |
| `MODEL_INPUT_COST_PER_MILLION_TOKENS` | Unset | Optional input rate used for estimated per-run cost. |
| `MODEL_OUTPUT_COST_PER_MILLION_TOKENS` | Unset | Optional output rate; cost is reported only when both rates and provider usage are available. |
| `ENABLE_SUBAGENTS` | `true` | Enables bounded `delegate_task` investigation by read-only child agents. |
| `MAX_SUBAGENTS_PER_RUN` | `2` | Maximum child runs per parent task. |
| `MAX_SUBAGENT_ITERATIONS` | `8` | Maximum planning iterations for each child. |
| `MAX_SUBAGENT_TOOL_CALLS` | `12` | Maximum tool calls for each child. |
| `MAX_SUBAGENT_TASK_CHARS` | `2000` | Maximum size of a delegated task description. |
| `LOG_LEVEL` | `INFO` | Log verbosity. |
| `LOG_FILE` | Unset | Optional rotating JSON-lines log file (10 MB, three backups). |
| `LOG_PAYLOADS` | `false` | Explicitly allows prompts, tool arguments/results, and MCP payloads in debug logs. |
| `COMMAND_TIMEOUT` | `120` seconds | Default command timeout. |
| `TOOL_TIMEOUT` | `120` seconds | Default MCP request timeout. |

Logs are structured JSON events. Model and tool payloads are omitted even at DEBUG level unless `LOG_PAYLOADS=true`; credential-like fields such as API keys, authorization values, passwords, and tokens are also redacted by key name. Provider-reported token usage, model-call latency, tool latency, and total run duration are aggregated in the run result and structured completion event.

The default context budget is a conservative estimate (serialized characters divided by three), not an exact tokenizer count. Planner tool schemas reserve part of the same budget. Pass a provider-specific `token_counter` to `ContextManager` when exact token accounting is available. Old history is compacted into bounded, labeled notes; the original task and recent tool-call/result pairs are retained.

Boolean settings accept `1`, `true`, `yes`, or `on` as enabled. Any other explicit value disables the setting. Restart the relevant process after changing its environment or `.env` values.

Each agent run writes planner/evaluator responses and tool results to a temporary JSONL task journal. After a tool failure, newly recorded journal entries are fed back into the agent; final synthesis rereads the full journal so earlier answers and successful results are retained. The temporary file is deleted when `AgentLoop.run()` returns. `LOG_FILE` is a separate, optional diagnostic log.

## Workspace Safety

Filesystem tools resolve paths against the configured workspace. Reads, writes, edits, searches, and deletes are restricted to that workspace. The workspace root itself cannot be deleted; deleting a directory removes its contents recursively, and symbolic links cannot be deleted through the delete tool.

Deletion is enabled by default. To turn it off, set `ALLOW_DELETE=false` before starting the MCP server. Git writes are disabled by default. Shell and network tools are enabled by default; only use this agent with a workspace and task scope you trust.

## Available Tool Groups

- **Filesystem:** read one or multiple files, write, edit, search, list, and delete files or directories.
- **Git:** inspect status, diffs, and history; selected Git mutations are permission-controlled.
- **Shell and processes:** execute commands and manage launched processes.
- **Browser:** navigate, interact with pages, inspect console/network activity, and capture screenshots.
- **Documents and images:** extract or convert document, PDF, HTML, and image content.
- **Testing and diagnostics:** run tests, linters, formatters, type checks, builds, and diagnostic collection.

The browser tools use Playwright's async API with Chromium. Install its browser binary if browser launch reports that Chromium is missing:

```powershell
python -m playwright install chromium
```

For a live-page task, agents should launch the browser, navigate with `browser_goto`, use `browser_wait` for client-rendered content, inspect with `browser_page_snapshot`, then pass a returned role selector (for example `role=button[name="Continue"]`) to an action tool. If navigation or an action appears to fail, check `browser_get_failed_requests`, `browser_get_responses`, and `browser_get_console`. Request and response history is retained across tabs and bounded to the most recent 1,000 events per category.

The browser and web tools serve different purposes: `web_search` discovers public information, while browser tools open and interact with a live page. Static extraction tools (`web_open`, `web_extract_text`, `web_extract_links`) are lighter alternatives when interaction is not needed.

For advanced agents, consider removing or exposing on demand:

- `quote_shell_argument`, `build_shell_command`, and `parse_shell_command` if `inspect_shell_command` plus `execute_command` already cover your preferred shell workflow.
- `browser_double_click` and `browser_type` for agents that can use `browser_click` and `browser_fill`; keep them only for UI tasks that specifically need those behaviors.
- Duplicate screenshot workflows: use `browser_screenshot` for web pages and `capture_screen` / `capture_region` only for desktop applications.
- The document/image conversion tools when a deployment does not process those file types.

Keep the focused file operations, Git inspection, testing, and diagnostics tools. They provide distinct capabilities that general shell access cannot return as structured, verifiable results. Prefer server-side task profiles or filtered tool discovery over permanently deleting tools needed by some workloads.

## Checks and Troubleshooting

With the MCP server running, list the tools currently exposed by its endpoint:

```powershell
python .\test\test_registry.py
```

The script prints discovered tool schemas and verifies that the file-batch and browser workflow tools are exposed. If tools are missing, confirm `MCP_URL`, restart the MCP server after code or configuration changes, and run the check again.

Run the focused pytest suite:

```powershell
python -m pytest -q test/tests
python -m pytest -q test/test_browser_tools.py
```

If the agent cannot connect, confirm the MCP server is running and that `MCP_URL` points to it. If model requests fail, check that Ollama is running with the selected model or that `NVIDIA_API_KEY` is configured.

`test_agent_loop.py` is a development harness with a task defined in the file, not a general command-line interface. It can execute real tools against its configured workspace; review its task and workspace before running it.

## Deploy with Prefect Cloud

`prefect_flow.py` exposes `run_agent(task, workspace_path=None)` as a Prefect flow. Each run creates a new model client, MCP client, tool registry, and agent loop, then returns a compact execution summary. The existing `test_agent_loop.py` remains a local development script.

This deployment definition uses Prefect Managed infrastructure and clones this repository for each flow run. Push the repository to a Git remote, then configure the deployment environment and create the work pool:

```powershell
python -m pip install -r requirements.txt
prefect cloud login
$env:PREFECT_SOURCE_URL = "https://github.com/OWNER/REPOSITORY.git"
prefect work-pool create local-agent-managed --type prefect:managed
prefect deploy --name agent-task
```

For a private Git repository, configure a Prefect GitHub Credentials block and reference it as the `credentials` value in the `git_clone` step in `prefect.yaml`. Configure runtime environment variables in the work pool/deployment, keeping credentials in Prefect secrets:

| Variable | Purpose |
| --- | --- |
| `AGENT_WORKSPACE` | Existing workspace directory available to the flow runtime. Defaults to the checked-out repository directory. |
| `MCP_URL` | HTTP MCP endpoint reachable from the flow runtime. |
| `MODEL_PROVIDER` | `nvidia` or `ollama`; defaults to `nvidia`. |
| `NVIDIA_API_KEY` | Required when `MODEL_PROVIDER=nvidia`. |
| `NVIDIA_MODEL` / `NVIDIA_ENDPOINT` | Optional NVIDIA model and API endpoint overrides. |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | Ollama URL and model; the service must be reachable from the flow runtime. |
| `TAVILY_API_KEY` | Required when using the Tavily-backed web search tool. |

The MCP tools server is a separate service and must have the target workspace mounted locally. The flow runtime also needs an existing corresponding `AGENT_WORKSPACE` directory with the same relative layout, because it performs path validation before tool calls. For containerized/private-network deployment, configure the MCP server with `MCP_HOST=0.0.0.0` and `MCP_PORT` as needed; keep it behind a private network or authenticated proxy because this server does not provide authentication. Prefect Managed runs do not provide a persistent custom workspace mount, so use a worker/work pool with a mounted workspace when tasks need to modify arbitrary repositories or persist edits.

For unattended runs, set conservative tool permissions such as `ALLOW_DELETE=false`, `ALLOW_GIT_WRITE=false`, and `ALLOW_SHELL=false` unless the task specifically requires them. The current defaults enable deletion and shell execution.

### FastMCP Server Hosting

If the hosting platform runs `fastmcp inspect` to build an MCP server, use `server.py:mcp` as its entry point. This re-exports the existing FastMCP instance from `fastmcp_tools_http.py`; it is not a Prefect flow. The Prefect deployment entry point remains `prefect_flow.py:run_agent` and must be deployed through Prefect.

### Vercel Hosting

`app.py` exposes the full FastMCP server as a stateless ASGI application for Vercel's Python Functions. Connect AgentStatus to `https://<your-vercel-domain>/mcp` and set its monitor authentication to `None`. In Vercel project settings, ensure Deployment Protection does not require login for the production endpoint. Vercel automatically detects `app.py` as the Python ASGI entry point and installs `requirements.txt`.

This publishes the complete tool set without authentication, including shell execution and filesystem write/delete operations. Any internet user who can reach the endpoint can invoke them; there is no way to limit access to "legitimate users" when the endpoint has no authentication. The Vercel workspace is deployment-local and ephemeral: file changes are not a durable shared workspace, and browser/process state is not guaranteed between requests. Function time limits also apply. Do not add API keys or other secrets to Vercel environment variables for this public deployment. Use a persistent container host with authentication for real workspace edits and long-running tasks.