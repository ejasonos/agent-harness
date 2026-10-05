from __future__ import annotations

from typing import Any

from workspace.manager import WorkspaceManager
from tools.testing.runner import run_process


def detect_typecheck_command(workspace: WorkspaceManager) -> str:
    root = workspace.root

    if (root / "tsconfig.json").exists():
        return "npx tsc --noEmit"

    if (root / "pyrightconfig.json").exists():
        return "pyright"

    if (root / "mypy.ini").exists():
        return "mypy ."

    if (root / "pyproject.toml").exists():
        return "mypy ."

    if (root / "go.mod").exists():
        return "go vet ./..."

    if (root / "Cargo.toml").exists():
        return "cargo check"

    raise RuntimeError(
        "Could not automatically determine a type-check command."
    )


def run_typecheck(
    workspace: WorkspaceManager,
    command: str | None = None,
    *,
    timeout: int = 120,
) -> dict[str, Any]:
    """Run static type checking."""

    resolved_command = command or detect_typecheck_command(workspace)

    result = run_process(
        workspace,
        resolved_command,
        timeout=timeout,
    )

    result["operation"] = "typecheck"

    return result