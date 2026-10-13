# Local Agent

Local Agent is a model-directed AI agent. It connects a model to an HTTP MCP server that exposes workspace, Git, shell, browser, conversion, screenshot, testing, and math tools. Use the Python `AgentLoop` API to submit tasks and inspect tool activity.

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

The default provider is Ollama, using `gemma4:e4b` at `http://127.0.0.1:11434`. Install Ollama for Windows from [ollama.com/download](https://ollama.com/download/windows), then open a new terminal and download the configured model:

```powershell
ollama pull gemma4:e4b
ollama list
```

The Ollama desktop app normally starts its local service. If the service is not running, start it in a separate terminal with `ollama serve`. Confirm the model responds with `ollama run gemma4:e4b` and exit the model prompt with `/bye`.

The development runner, `test_agent_loop.py`, uses `OllamaClient` and does not require an NVIDIA key. To use NVIDIA instead, configure `NVIDIA_API_KEY` and explicitly construct and pass an `NVIDIAClient`; the default NVIDIA model is `meta/llama-3.2-11b-vision-instruct`.

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

from agent.loop import AgentLoop
from config.settings import load_settings
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry
from model.ollama_client import OllamaClient
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
model = OllamaClient(
	model=settings.model,
	base_url=settings.ollama_host,
	timeout=600,
)
agent = AgentLoop(
	model=model,
	mcp_client=mcp_client,
	registry=ToolRegistry(mcp_client),
	workspace=workspace,
)

result = agent.run("Explain the project structure")
print(result.final_answer)
if result.error:
	print(result.error)
```

Replace the example task and workspace path with your own. `result.tool_executions` contains each tool's name, reason, arguments, result, and status. To use NVIDIA instead of Ollama, construct `NVIDIAClient` with `settings.nvidia_model`, `settings.nvidia_api_key`, and `settings.nvidia_endpoint`.

The MCP server and agent program must use the same workspace directory. The server uses `AGENT_WORKSPACE` or its current working directory; `load_settings(workspace_path)` configures the agent's local workspace.

### How Ollama Is Wired In

For a local Ollama model, the project already contains the client and configuration path; the agent loop and MCP server do not need provider-specific changes:

- `config/defaults.py` sets the default model name to `gemma4:e4b`.
- `config/settings.py` reads `OLLAMA_MODEL` and `OLLAMA_HOST` from the environment and supplies those values to the client.
- `model/ollama_client.py` sends chat requests to Ollama's `/api/chat` endpoint and exposes the available-model health/list checks. It uses the `requests` dependency already in `requirements.txt`; no Ollama Python package is needed.
- `test_agent_loop.py` constructs `OllamaClient` from those settings and passes it to `AgentLoop`. This is the provider selection point for that runner.
- `agent/planner.py` calls the provider through the shared `chat()` interface. It can stay unchanged when switching between the Ollama and NVIDIA clients.
- `fastmcp_tools_http.py` is a separate tools server. It does not call the model and does not need Ollama configuration.

Set these values in the project `.env` file when using a different model or Ollama host:

```dotenv
OLLAMA_MODEL=gemma4:e4b
OLLAMA_HOST=http://127.0.0.1:11434
```

`OLLAMA_HOST` is the Ollama server's base URL, not the `/api/chat` endpoint. `OLLAMA_MODEL` must match a model shown by `ollama list` and should support tool calling if you expect the agent to use MCP tools. After changing `.env`, restart the agent process. Ollama handles local inference; web search is a separate tool and still requires network access and `TAVILY_API_KEY`.

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
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=gemma4:e4b

# Optional NVIDIA API provider
NVIDIA_API_KEY=your-key
NVIDIA_ENDPOINT=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.2-11b-vision-instruct

# Required for web_search
TAVILY_API_KEY=your-tavily-key

# Structured agent/model/MCP trace output
LOG_LEVEL=DEBUG
LOG_FILE=logs/agent.jsonl

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
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API URL. |
| `OLLAMA_MODEL` | `gemma4:e4b` | Default local model. |
| `NVIDIA_API_KEY` | Unset | Required when using NVIDIA API. |
| `NVIDIA_ENDPOINT` | `https://integrate.api.nvidia.com/v1` | NVIDIA-compatible API endpoint. |
| `NVIDIA_MODEL` | `meta/llama-3.2-11b-vision-instruct` | Default NVIDIA model. |
| `TAVILY_API_KEY` | Unset | Required by the Tavily-backed `web_search` tool. |
| `ALLOW_DELETE` | `true` | Enables delete operations. Set to `false` to disable them. |
| `ALLOW_GIT_WRITE` | `false` | Enables Git mutations. |
| `ALLOW_SHELL` | `true` | Enables shell and process tools. |
| `ALLOW_NETWORK` | `true` | Enables network-dependent tools. |
| `MAX_ITERATIONS` | `30` | Maximum agent planning iterations. |
| `MAX_TOOL_CALLS` | `100` | Maximum tool calls per run. |
| `LOG_LEVEL` | `INFO` | Log verbosity; use `DEBUG` for full model and MCP payloads. |
| `LOG_FILE` | Unset | Optional rotating JSON-lines log file (10 MB, three backups). |
| `COMMAND_TIMEOUT` | `120` seconds | Default command timeout. |
| `TOOL_TIMEOUT` | `120` seconds | Default MCP request timeout. |

Logs are structured JSON events. DEBUG payload events can include prompts, model responses, tool arguments, and workspace data; protect log files accordingly. Credential-like fields such as API keys, authorization values, passwords, and tokens are redacted by key name.

Boolean settings accept `1`, `true`, `yes`, or `on` as enabled. Any other explicit value disables the setting. Restart the relevant process after changing its environment or `.env` values.

Each agent run writes planner/evaluator responses and tool results to a temporary JSONL task journal. After a tool failure, newly recorded journal entries are fed back into the agent; final synthesis rereads the full journal so earlier answers and successful results are retained. The temporary file is deleted when `AgentLoop.run()` returns. `LOG_FILE` is a separate, optional diagnostic log.

## Workspace Safety

Filesystem tools resolve paths against the configured workspace. Reads, writes, edits, searches, and deletes are restricted to that workspace. The workspace root itself cannot be deleted; deleting a directory removes its contents recursively, and symbolic links cannot be deleted through the delete tool.

Deletion is enabled by default. To turn it off, set `ALLOW_DELETE=false` before starting the MCP server. Git writes are disabled by default. Shell and network tools are enabled by default; only use this agent with a workspace and task scope you trust.

## Available Tool Groups

- **Filesystem:** read, write, edit, search, list, and delete files or directories.
- **Project:** detect projects and frameworks, discover project commands.
- **Git:** inspect status, diffs, and history; selected Git mutations are permission-controlled.
- **Shell and processes:** execute commands and manage launched processes.
- **Browser:** navigate, interact with pages, inspect console/network activity, and capture screenshots.
- **Documents and images:** extract or convert document, PDF, HTML, and image content.
- **Testing and diagnostics:** run tests, linters, formatters, type checks, builds, and diagnostic collection.
- **Math:** addition, subtraction, multiplication, and division.

The browser tools use Playwright Chromium. Install its browser binary if browser launch reports that Chromium is missing:

```powershell
python -m playwright install chromium
```

## Checks and Troubleshooting

With the MCP server running, list the tools currently exposed by its endpoint:

```powershell
python test_registry.py
```

The script prints the endpoint and discovered tool schemas, and fails if the four math tools are missing. If tools are missing, confirm `MCP_URL`, restart the MCP server after code or configuration changes, and run the check again.

Run the focused pytest suite:

```powershell
python -m pytest -q test/tests
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
| `MODEL_PROVIDER` | `nvidia` or `ollama`; defaults to `ollama`. |
| `NVIDIA_API_KEY` | Required when `MODEL_PROVIDER=nvidia`. |
| `NVIDIA_MODEL` / `NVIDIA_ENDPOINT` | Optional NVIDIA model and API endpoint overrides. |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | Ollama URL and model; the service must be reachable from the flow runtime. |
| `TAVILY_API_KEY` | Required when using the Tavily-backed web search tool. |

The MCP tools server is a separate service and must have the target workspace mounted locally. The flow runtime also needs an existing corresponding `AGENT_WORKSPACE` directory with the same relative layout, because it performs path validation before tool calls. For containerized/private-network deployment, configure the MCP server with `MCP_HOST=0.0.0.0` and `MCP_PORT` as needed; keep it behind a private network or authenticated proxy because this server does not provide authentication. Prefect Managed runs do not provide a persistent custom workspace mount, so use a worker/work pool with a mounted workspace when tasks need to modify arbitrary repositories or persist edits.

For unattended runs, set conservative tool permissions such as `ALLOW_DELETE=false`, `ALLOW_GIT_WRITE=false`, and `ALLOW_SHELL=false` unless the task specifically requires them. The current defaults enable deletion and shell execution.