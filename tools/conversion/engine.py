from __future__ import annotations

from pathlib import Path
from typing import Any

from workspace.manager import WorkspaceManager


def resolve_input(
    workspace: WorkspaceManager,
    path: str,
) -> Path:
    resolved = workspace.read_path(path)

    if not resolved.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    if not resolved.is_file():
        raise IsADirectoryError(
            f"Input path is not a file: {path}"
        )

    return resolved


def resolve_output(
    workspace: WorkspaceManager,
    path: str,
) -> Path:
    resolved = workspace.write_path(path)

    if resolved.exists() and resolved.is_dir():
        raise IsADirectoryError(
            f"Output path is a directory: {path}"
        )

    resolved.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return resolved


def require_dependency(
    module_name: str,
    install_name: str | None = None,
) -> Any:
    try:
        return __import__(
            module_name
        )
    except ImportError as exc:
        package = install_name or module_name

        raise RuntimeError(
            f"Conversion feature requires '{package}'. "
            f"Install it with: pip install {package}"
        ) from exc