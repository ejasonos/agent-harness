from __future__ import annotations

import shutil
from pathlib import Path

from workspace.manager import WorkspaceManager


def delete_file(
    workspace: WorkspaceManager,
    path: str,
) -> str:
    """Delete a file or directory inside the agent workspace."""

    requested_path = Path(path)
    if not requested_path.is_absolute():
        requested_path = workspace.root / requested_path

    if requested_path.is_symlink():
        raise PermissionError(
            "Deleting symbolic links is not supported."
        )

    target = workspace.delete_path(path)

    if target == workspace.root:
        raise PermissionError(
            "Deleting the workspace root is not allowed."
        )

    if not target.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if target.is_dir():
        shutil.rmtree(target)
        return f"Successfully deleted directory: {path}"

    if target.is_file():
        target.unlink()
        return f"Successfully deleted file: {path}"

    raise ValueError(
        f"Path is not a regular file or directory: {path}"
    )