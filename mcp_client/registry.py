from __future__ import annotations

from copy import deepcopy
from typing import Any

from mcp_client.client import MCPClient
from mcp_client.schemas import MCPTool, ToolPolicy


class ToolRegistry:
    """
    Discovers, normalizes, and exposes MCP tools to the agent.

    The registry is responsible for:
    - discovering available MCP tools
    - normalizing their schemas
    - attaching execution policies
    - exposing all tools to the model

    Tool policies do NOT hide tools from the model.
    The model always receives the complete available tool set.
    Policies are enforced later by the agent runtime before execution.
    """

    def __init__(
        self,
        client: MCPClient,
    ) -> None:
        self.client = client
        self._tools: dict[str, MCPTool] = {}

    # =============================================================
    # DISCOVERY
    # =============================================================

    def discover(self) -> list[MCPTool]:
        """
        Discover tools from the MCP server and normalize them.

        Every discovered tool receives a default ToolPolicy.

        Policies are metadata only at this layer. All discovered
        tools remain available to the model.
        """

        raw_tools = self.client.list_tools()

        self._tools.clear()

        for raw_tool in raw_tools:
            name = self._extract_tool_name(raw_tool)

            if not name:
                continue

            description = self._extract_tool_description(
                raw_tool
            )

            input_schema = self._extract_input_schema(
                raw_tool
            )

            policy = self._default_policy_for_tool(
                name=name,
            )

            tool = MCPTool(
                name=name,
                description=description,
                input_schema=input_schema,
                policy=policy,
            )

            self._tools[name] = tool

        return self.all()

    # =============================================================
    # TOOL LOOKUP
    # =============================================================

    def get(
        self,
        name: str,
    ) -> MCPTool | None:
        """
        Return a registered tool by name.
        """

        return self._tools.get(name)

    def has(
        self,
        name: str,
    ) -> bool:
        """
        Return True if a tool is registered.
        """

        return name in self._tools

    def all(self) -> list[MCPTool]:
        """
        Return every registered tool.

        Policies do not filter this list.
        """

        return list(self._tools.values())

    # =============================================================
    # POLICY
    # =============================================================

    def _default_policy_for_tool(
        self,
        name: str,
    ) -> ToolPolicy:
        """
        Return the default execution policy for a tool.

        Policies describe contextual knowledge that should exist
        before executing a tool.

        IMPORTANT:

        Policies do not determine whether a tool is shown to the
        model. Every discovered tool is exposed to the model.

        The agent runtime is responsible for enforcing these
        prerequisites before execution.
        """

        # ---------------------------------------------------------
        # DISCOVERY TOOLS
        #
        # These tools establish workspace/project knowledge and
        # therefore do not require prior workspace knowledge.
        # ---------------------------------------------------------

        discovery_tools = {
            "directory_tree",
            "list_directory",
            "list_files",
            "workspace_tree",
            "project_tree",
            "detect_project",
            "project_info",
            "project_commands",
            "detect_framework",
            "git_status",
            "git_diff",
            "git_history",
        }

        # ---------------------------------------------------------
        # INSPECTION TOOLS
        #
        # These operate on an already-known workspace.
        # ---------------------------------------------------------

        inspection_tools = {
            "read_file",
            "read_files",
            "inspect_file",
            "search_files",
            "grep",
            "find_in_files",

            # Document / image inspection
            "extract_text",
            "extract_document_text",
            "extract_pdf_text",
            "image_info",
            "html_to_text",
        }

        browser_inspection_tools = {
            "browser_page_text",
            "browser_element_text",
            "browser_element_attribute",
            "browser_element_exists",
            "browser_page_html",
            "browser_current_page",
            "browser_get_console",
            "browser_get_requests",
            "browser_get_responses",
        }

        # ---------------------------------------------------------
        # FILE / DOCUMENT MUTATION TOOLS
        #
        # These require workspace knowledge and knowledge of the
        # target before mutation.
        #
        # They also require post-execution verification.
        # ---------------------------------------------------------

        mutation_tools = {
            "write_file",
            "edit_file",
            "modify_file",
            "patch_file",
            "delete_file",
            "move_file",
            "rename_file",

            # Transformation tools
            "crop_image",
            "resize_image",
            "convert_document",
            "convert_image",
            "download_file",
            "web_download",
            "render_pdf_pages",
            "html_to_pdf",
            "create_pdf",
            "capture_screen",
            "capture_region",
        }

        # ---------------------------------------------------------
        # SHELL / PROCESS TOOLS
        #
        # Shell operations require knowledge of the workspace but
        # do not necessarily require a previously inspected target.
        # ---------------------------------------------------------

        shell_tools = {
            "execute_command",
            "start_process",
            "stop_process",
            "remove_process",
            "process_status",
        }

        # ---------------------------------------------------------
        # TESTING / VERIFICATION TOOLS
        #
        # These can be used after workspace discovery. They may
        # themselves reveal useful information about the project,
        # so we do not require target_inspected here.
        # ---------------------------------------------------------

        testing_tools = {
            "run_tests",
            "run_test_file",
            "run_linter",
            "run_formatter_check",
            "run_typecheck",
            "run_build",
        }

        # ---------------------------------------------------------
        # FORMATTER
        #
        # Formatting changes files, so it follows mutation policy.
        # ---------------------------------------------------------

        if name == "run_formatter":
            return ToolPolicy(
                prerequisites=[
                    "workspace_discovered",
                    "target_inspected",
                ],
                verify_after=True,
            )

        # ---------------------------------------------------------
        # GIT MUTATION TOOLS
        #
        # Git mutations require workspace knowledge and should be
        # followed by verification.
        # ---------------------------------------------------------

        git_mutation_tools = {
            "git_add",
            "git_unstage",
            "git_commit",
            "git_create_branch",
            "git_switch_branch",
            "git_delete_branch",
            "git_restore",
        }

        # ---------------------------------------------------------
        # BROWSER ACTION TOOLS
        #
        # Browser actions operate on browser state rather than
        # workspace knowledge.
        # ---------------------------------------------------------

        browser_action_tools = {
            "browser_launch",
            "browser_close",
            "browser_status",
            "browser_goto",
            "browser_click",
            "browser_double_click",
            "browser_type",
            "browser_fill",
            "browser_select",
            "browser_press",
            "browser_check",
            "browser_uncheck",
            "browser_hover",
            "browser_scroll",
            "browser_set_viewport",
            "browser_back",
            "browser_forward",
            "browser_reload",
            "browser_clear_console",
            "browser_clear_network",
        }

        # ---------------------------------------------------------
        # SCREENSHOT TOOLS
        # ---------------------------------------------------------

        screenshot_tools = {
            "take_screenshot",
            "screenshot",
            "capture_screenshot",
            "browser_screenshot",
        }

        # ---------------------------------------------------------
        # DIAGNOSTIC TOOLS
        # ---------------------------------------------------------

        diagnostics_tools = {
            "collect_diagnostics",
            "parse_diagnostics",
        }

        # ---------------------------------------------------------
        # POLICY RESOLUTION
        # ---------------------------------------------------------

        if name in discovery_tools:
            return ToolPolicy()

        if name in inspection_tools:
            return ToolPolicy(
                prerequisites=[
                    "workspace_discovered",
                ],
            )

        if name in browser_inspection_tools:
            return ToolPolicy()

        if name in mutation_tools:
            return ToolPolicy(
                prerequisites=[
                    "workspace_discovered",
                    "target_inspected",
                ],
                verify_after=name != "delete_file",
            )

        if name in shell_tools:
            return ToolPolicy(
                prerequisites=[
                    "workspace_discovered",
                ],
            )

        if name in testing_tools:
            return ToolPolicy(
                prerequisites=[
                    "workspace_discovered",
                ],
            )

        if name in git_mutation_tools:
            return ToolPolicy(
                prerequisites=[
                    "workspace_discovered",
                ],
                verify_after=True,
            )

        if name in browser_action_tools:
            return ToolPolicy()

        if name in screenshot_tools:
            return ToolPolicy()

        if name in diagnostics_tools:
            return ToolPolicy()

        # ---------------------------------------------------------
        # UNKNOWN TOOLS
        #
        # Unknown tools remain unrestricted by default.
        #
        # This is important because the registry should not
        # accidentally impose assumptions on newly added tools.
        # ---------------------------------------------------------

        return ToolPolicy()

    # =============================================================
    # MODEL TOOL DEFINITIONS
    # =============================================================

    def to_ollama_tools(
        self,
    ) -> list[dict[str, Any]]:
        """
        Convert every registered tool into Ollama-compatible
        tool definitions.

        Policy metadata is intentionally not included in the model
        schema. The model needs the tool's capability and interface,
        while policy enforcement remains a runtime responsibility.
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self._input_schema_with_reason(
                        tool.input_schema
                    ),
                },
            }
            for tool in self.all()
        ]

    def to_openai_tools(
        self,
    ) -> list[dict[str, Any]]:
        """
        Convert every registered tool into OpenAI-compatible
        tool definitions.

        All tools are exposed regardless of their current policy
        prerequisites.
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self._input_schema_with_reason(
                        tool.input_schema
                    ),
                },
            }
            for tool in self.all()
        ]

    @staticmethod
    def _input_schema_with_reason(
        input_schema: dict[str, Any],
    ) -> dict[str, Any]:
        schema = deepcopy(input_schema)
        properties = schema.setdefault("properties", {})
        properties["reason_for_tool"] = {
            "type": "string",
            "description": "Briefly explain why this tool call is needed.",
        }
        required = schema.setdefault("required", [])
        if "reason_for_tool" not in required:
            required.append("reason_for_tool")
        return schema

    # =============================================================
    # RAW TOOL NORMALIZATION
    # =============================================================

    def _extract_tool_name(
        self,
        raw_tool: Any,
    ) -> str | None:
        """
        Extract a tool name from an MCP tool representation.
        """

        if isinstance(raw_tool, dict):
            name = raw_tool.get("name")

            if isinstance(name, str):
                return name

            return None

        name = getattr(
            raw_tool,
            "name",
            None,
        )

        if isinstance(name, str):
            return name

        return None

    def _extract_tool_description(
        self,
        raw_tool: Any,
    ) -> str:
        """
        Extract a tool description from an MCP tool representation.
        """

        if isinstance(raw_tool, dict):
            description = raw_tool.get(
                "description",
                "",
            )

            return (
                str(description)
                if description is not None
                else ""
            )

        description = getattr(
            raw_tool,
            "description",
            "",
        )

        return (
            str(description)
            if description is not None
            else ""
        )

    def _extract_input_schema(
        self,
        raw_tool: Any,
    ) -> dict[str, Any]:
        """
        Extract the input schema from an MCP tool representation.
        """

        if isinstance(raw_tool, dict):
            schema = raw_tool.get(
                "inputSchema"
            )

            if schema is None:
                schema = raw_tool.get(
                    "input_schema"
                )

            if isinstance(schema, dict):
                return schema

            return {}

        schema = getattr(
            raw_tool,
            "inputSchema",
            None,
        )

        if schema is None:
            schema = getattr(
                raw_tool,
                "input_schema",
                None,
            )

        if isinstance(schema, dict):
            return schema

        return {}