from __future__ import annotations

import subprocess
from pathlib import Path

from workspace.manager import WorkspaceManager


def _run_git(
    workspace: WorkspaceManager,
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    """
    Execute a Git command from the workspace root.
    """

    root = Path(workspace.root)

    if not root.exists():
        raise FileNotFoundError(
            "Workspace root does not exist."
        )

    if not root.is_dir():
        raise NotADirectoryError(
            "Workspace root is not a directory."
        )

    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Git executable was not found."
        ) from exc


def _require_success(
    result: subprocess.CompletedProcess[str],
    operation: str,
) -> None:
    if result.returncode == 0:
        return

    error = result.stderr.strip()

    raise RuntimeError(
        f"Git {operation} failed"
        + (f": {error}" if error else ".")
    )


def git_status(
    workspace: WorkspaceManager,
) -> dict[str, object]:
    """
    Return the current Git repository status.
    """

    result = _run_git(
        workspace,
        [
            "status",
            "--short",
            "--branch",
        ],
    )

    _require_success(
        result,
        "status",
    )

    lines = [
        line
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    branch = ""
    changes: list[str] = []

    for line in lines:
        if line.startswith("## "):
            branch = line[3:].strip()
        else:
            changes.append(line)

    return {
        "is_repository": True,
        "branch": branch,
        "clean": len(changes) == 0,
        "changes": changes,
        "change_count": len(changes),
        "raw": result.stdout.strip(),
    }