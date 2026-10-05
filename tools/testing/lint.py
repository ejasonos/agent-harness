from __future__ import annotations

from pathlib import Path
from typing import Any

from workspace.manager import WorkspaceManager
from tools.testing.runner import run_process


def detect_lint_command(workspace: WorkspaceManager) -> str:
    root = workspace.root

    if (root / "package.json").exists():
        return "npm run lint"

    if (root / "pyproject.toml").exists():
        return "ruff check ."

    if (root / "go.mod").exists():
        return "go vet ./..."

    if (root / "Cargo.toml").exists():
        return "cargo clippy --all-targets --all-features -- -D warnings"

    raise RuntimeError(
        "Could not automatically determine a lint command."
    )


def run_linter(
    workspace: WorkspaceManager,
    command: str | None = None,
    *,
    timeout: int = 120,
) -> dict[str, Any]:
    """Run the project's linter."""

    resolved_command = command or detect_lint_command(workspace)

    result = run_process(
        workspace,
        resolved_command,
        timeout=timeout,
    )

    result["operation"] = "lint"

    return result


def run_formatter_check(
    workspace: WorkspaceManager,
    command: str | None = None,
    *,
    timeout: int = 120,
) -> dict[str, Any]:
    """Check whether project formatting is valid."""

    root = workspace.root

    if command:
        resolved_command = command
    elif (root / "package.json").exists():
        resolved_command = "npm run format:check"
    elif (root / "pyproject.toml").exists():
        resolved_command = "ruff format --check ."
    elif (root / "Cargo.toml").exists():
        resolved_command = "cargo fmt --check"
    else:
        raise RuntimeError(
            "Could not automatically determine a formatter check command."
        )

    result = run_process(
        workspace,
        resolved_command,
        timeout=timeout,
    )

    result["operation"] = "formatter_check"

    return result


def run_formatter(
    workspace: WorkspaceManager,
    command: str | None = None,
    *,
    timeout: int = 120,
) -> dict[str, Any]:
    """Run the project's formatter."""

    root = workspace.root

    if command:
        resolved_command = command
    elif (root / "package.json").exists():
        resolved_command = "npm run format"
    elif (root / "pyproject.toml").exists():
        resolved_command = "ruff format ."
    elif (root / "Cargo.toml").exists():
        resolved_command = "cargo fmt"
    else:
        raise RuntimeError(
            "Could not automatically determine a formatter command."
        )

    result = run_process(
        workspace,
        resolved_command,
        timeout=timeout,
    )

    result["operation"] = "format"

    return result