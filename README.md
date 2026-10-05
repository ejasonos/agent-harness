# Local Agent

Local Agent is a model-directed coding assistant. It connects a model to an HTTP MCP server that exposes workspace, Git, shell, browser, conversion, screenshot, testing, and math tools. The Streamlit interface is the recommended way to submit tasks and review tool activity.

## Quick Start

Run commands from the project directory. The MCP server and the Streamlit interface run as separate processes, so keep two terminals open.

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

Alternatively, configure an NVIDIA API key and select **NVIDIA API** in the app. The default NVIDIA model is `meta/llama-3.2-11b-vision-instruct`.

### 3. Start the MCP server

In the first terminal, from the project directory:

```powershell
python fastmcp_tools_http.py
```

The server listens on `http://127.0.0.1:8082/mcp` by default. Leave this process running while using the app.

### 4. Start the user interface

In a second terminal, also from the project directory:

```powershell
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. Choose a workspace, MCP endpoint, and model provider in the sidebar, then enter a task in the chat box. Review the tool activity and results in the assistant response.

The MCP server and the app must use the same workspace directory. The UI checks and applies its workspace setting locally; the server independently uses `AGENT_WORKSPACE` or the directory from which it was started.

### Use a different workspace

In each PowerShell terminal, set `AGENT_WORKSPACE` to the existing project directory before starting the process:

```powershell
$env:AGENT_WORKSPACE = "D:\work\my-project"
```

Then start `python fastmcp_tools_http.py` in one terminal and `streamlit run streamlit_app.py` in the other. PowerShell environment variables are scoped to their terminal, so set the value in both terminals. In the app sidebar, set **Agent workspace** to the same directory. Alternatively, put `AGENT_WORKSPACE=D:\work\my-project` in the project `.env` file; both processes load it at startup. Restart the MCP server after changing the workspace.

## Configuration

Both the server and Streamlit app load a `.env` file from the project directory. Environment variables override the defaults. For example:

```dotenv
AGENT_WORKSPACE=C:\path\to\your\project
MCP_URL=http://127.0.0.1:8082/mcp
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=gemma4:e4b

# Optional NVIDIA API provider
NVIDIA_API_KEY=your-key
NVIDIA_ENDPOINT=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.2-11b-vision-instruct

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

If the Streamlit app cannot connect, first confirm the MCP server is still running and that the URL in the sidebar points to it. If model requests fail, check that Ollama is running with the selected model or that `NVIDIA_API_KEY` is configured.

`test_agent_loop.py` is a development harness with a task defined in the file, not a general command-line interface. It can execute real tools against its configured workspace; review its task and workspace before running it.