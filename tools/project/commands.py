from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workspace.manager import WorkspaceManager


def _load_package_json(
    project_path: Path,
) -> dict[str, Any]:
    package_json = project_path / "package.json"

    if not package_json.exists():
        return {}

    try:
        data = json.loads(
            package_json.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def _node_commands(
    project_path: Path,
) -> dict[str, str]:
    package = _load_package_json(project_path)

    scripts = package.get("scripts", {})

    if not isinstance(scripts, dict):
        return {}

    commands: dict[str, str] = {}

    for name, command in scripts.items():
        if not isinstance(name, str):
            continue

        if not isinstance(command, str):
            continue

        commands[name] = f"npm run {name}"

    return commands


def _python_commands(
    project_path: Path,
) -> dict[str, str]:
    commands: dict[str, str] = {}

    if (project_path / "pytest.ini").exists():
        commands["test"] = "pytest"

    if (project_path / "pyproject.toml").exists():
        commands.setdefault(
            "test",
            "pytest",
        )

    if (
        (project_path / "requirements.txt").exists()
        or (project_path / "pyproject.toml").exists()
    ):
        commands.setdefault(
            "python",
            "python",
        )

    return commands


def project_commands(
    workspace: WorkspaceManager,
    path: str = ".",
) -> dict[str, object]:
    """
    Discover commands defined by the current project.

    Commands are discovered from project configuration rather than
    executed. This allows the agent to understand how the project
    should be built, tested, linted, or started.
    """

    project_path = workspace.read_path(path)

    if not project_path.exists():
        raise FileNotFoundError(
            f"Project path not found: {path}"
        )

    if not project_path.is_dir():
        raise NotADirectoryError(
            f"Project path is not a directory: {path}"
        )

    commands: dict[str, str] = {}

    commands.update(
        _node_commands(project_path)
    )

    python_commands = _python_commands(
        project_path
    )

    for name, command in python_commands.items():
        commands.setdefault(name, command)

    if (project_path / "go.mod").exists():
        commands.setdefault(
            "test",
            "go test ./...",
        )
        commands.setdefault(
            "build",
            "go build ./...",
        )

    if (project_path / "Cargo.toml").exists():
        commands.setdefault(
            "test",
            "cargo test",
        )
        commands.setdefault(
            "build",
            "cargo build",
        )

    return {
        "path": str(project_path),
        "commands": commands,
        "command_count": len(commands),
    }