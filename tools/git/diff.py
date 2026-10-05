from __future__ import annotations

import subprocess
from pathlib import Path

from workspace.manager import WorkspaceManager


def _run_git(
    workspace: WorkspaceManager,
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
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


def git_diff(
    workspace: WorkspaceManager,
    staged: bool = False,
    path: str | None = None,
) -> dict[str, object]:
    """
    Return the current Git diff.

    staged=False returns unstaged changes.

    staged=True returns changes currently staged for commit.

    An optional path can restrict the diff to a specific file
    or directory.
    """

    arguments = ["diff"]

    if staged:
        arguments.append("--cached")

    if path:
        arguments.extend(
            [
                "--",
                path,
            ]
        )

    result = _run_git(
        workspace,
        arguments,
    )

    _require_success(
        result,
        "diff",
    )

    diff_text = result.stdout

    return {
        "staged": staged,
        "path": path,
        "has_changes": bool(diff_text.strip()),
        "diff": diff_text,
    }