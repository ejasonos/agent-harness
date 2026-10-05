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

The default provider is Ollama, using `gemma4:e4b` at `http://127.0.0.1:11434`. Install and start Ollama separately, then make sure that model is available:

```powershell
ollama pull gemma4:e4b
```

Alternatively, configure `NVIDIA_API_KEY` and construct an `NVIDIAClient` in your Python program. The default NVIDIA model is `meta/llama-3.2-11b-vision-instruct`.

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
| `COMMAND_TIMEOUT` | `120` seconds | Default command timeout. |
| `TOOL_TIMEOUT` | `120` seconds | Default MCP request timeout. |

Boolean settings accept `1`, `true`, `yes`, or `on` as enabled. Any other explicit value disables the setting. Restart the relevant process after changing its environment or `.env` values.

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