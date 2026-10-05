from __future__ import annotations

from typing import Any

from workspace.manager import WorkspaceManager
from tools.testing.runner import run_process


def detect_build_command(workspace: WorkspaceManager) -> str:
    root = workspace.root

    if (root / "package.json").exists():
        return "npm run build"

    if (root / "pyproject.toml").exists():
        return "python -m build"

    if (root / "go.mod").exists():
        return "go build ./..."

    if (root / "Cargo.toml").exists():
        return "cargo build"

    if (root / "pom.xml").exists():
        return "mvn package"

    if (root / "gradlew").exists():
        return "./gradlew build"

    if (root / "gradlew.bat").exists():
        return "gradlew.bat build"

    raise RuntimeError(
        "Could not automatically determine a build command."
    )


def run_build(
    workspace: WorkspaceManager,
    command: str | None = None,
    *,
    timeout: int = 300,
) -> dict[str, Any]:
    """Build the project."""

    resolved_command = command or detect_build_command(workspace)

    result = run_process(
        workspace,
        resolved_command,
        timeout=timeout,
    )

    result["operation"] = "build"

    return result