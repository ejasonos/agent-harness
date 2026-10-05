from __future__ import annotations

import subprocess
from pathlib import Path

from workspace.manager import WorkspaceManager


def _run_git(
    workspace: WorkspaceManager,
    arguments: list[str],
    *,
    write: bool = False,
) -> subprocess.CompletedProcess[str]:
    if write:
        workspace.check_git_write()

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


def git_add(
    workspace: WorkspaceManager,
    paths: list[str],
) -> dict[str, object]:
    """
    Stage specific files or directories.
    """

    if not paths:
        raise ValueError(
            "At least one path is required."
        )

    arguments = [
        "add",
        "--",
        *paths,
    ]

    result = _run_git(
        workspace,
        arguments,
        write=True,
    )

    _require_success(
        result,
        "add",
    )

    return {
        "operation": "add",
        "paths": paths,
        "success": True,
        "output": result.stdout.strip(),
    }


def git_unstage(
    workspace: WorkspaceManager,
    paths: list[str],
) -> dict[str, object]:
    """
    Remove files from the Git staging area without
    deleting their working-tree changes.
    """

    if not paths:
        raise ValueError(
            "At least one path is required."
        )

    arguments = [
        "restore",
        "--staged",
        "--",
        *paths,
    ]

    result = _run_git(
        workspace,
        arguments,
        write=True,
    )

    _require_success(
        result,
        "restore --staged",
    )

    return {
        "operation": "unstage",
        "paths": paths,
        "success": True,
        "output": result.stdout.strip(),
    }


def git_commit(
    workspace: WorkspaceManager,
    message: str,
) -> dict[str, object]:
    """
    Create a Git commit using the supplied commit message.

    Only already-staged changes are committed.
    """

    message = message.strip()

    if not message:
        raise ValueError(
            "Commit message cannot be empty."
        )

    if len(message) > 500:
        raise ValueError(
            "Commit message must be <= 500 characters."
        )

    result = _run_git(
        workspace,
        [
            "commit",
            "-m",
            message,
        ],
        write=True,
    )

    _require_success(
        result,
        "commit",
    )

    return {
        "operation": "commit",
        "success": True,
        "message": message,
        "output": (
            result.stdout.strip()
            or result.stderr.strip()
        ),
    }


def git_create_branch(
    workspace: WorkspaceManager,
    branch: str,
    checkout: bool = False,
) -> dict[str, object]:
    """
    Create a new Git branch.

    checkout=True also switches the workspace to the new branch.
    """

    branch = branch.strip()

    if not branch:
        raise ValueError(
            "Branch name cannot be empty."
        )

    if branch.startswith("-"):
        raise ValueError(
            "Invalid branch name."
        )

    arguments = [
        "switch",
        "-c",
        branch,
    ]

    if not checkout:
        arguments = [
            "branch",
            branch,
        ]

    result = _run_git(
        workspace,
        arguments,
        write=True,
    )

    _require_success(
        result,
        "branch creation",
    )

    return {
        "operation": "create_branch",
        "branch": branch,
        "checked_out": checkout,
        "success": True,
        "output": result.stdout.strip(),
    }


def git_switch_branch(
    workspace: WorkspaceManager,
    branch: str,
) -> dict[str, object]:
    """
    Switch to an existing local Git branch.
    """

    branch = branch.strip()

    if not branch:
        raise ValueError(
            "Branch name cannot be empty."
        )

    if branch.startswith("-"):
        raise ValueError(
            "Invalid branch name."
        )

    result = _run_git(
        workspace,
        [
            "switch",
            branch,
        ],
        write=True,
    )

    _require_success(
        result,
        "switch",
    )

    return {
        "operation": "switch",
        "branch": branch,
        "success": True,
        "output": result.stdout.strip(),
    }


def git_delete_branch(
    workspace: WorkspaceManager,
    branch: str,
    force: bool = False,
) -> dict[str, object]:
    """
    Delete a local Git branch.

    force=False uses normal Git safety checks.
    """

    branch = branch.strip()

    if not branch:
        raise ValueError(
            "Branch name cannot be empty."
        )

    if branch.startswith("-"):
        raise ValueError(
            "Invalid branch name."
        )

    arguments = [
        "branch",
        "-D" if force else "-d",
        branch,
    ]

    result = _run_git(
        workspace,
        arguments,
        write=True,
    )

    _require_success(
        result,
        "branch deletion",
    )

    return {
        "operation": "delete_branch",
        "branch": branch,
        "force": force,
        "success": True,
        "output": result.stdout.strip(),
    }


def git_restore(
    workspace: WorkspaceManager,
    paths: list[str],
    staged: bool = False,
) -> dict[str, object]:
    """
    Restore files from Git.

    staged=False restores working-tree changes.

    staged=True restores the index while preserving
    working-tree changes.
    """

    if not paths:
        raise ValueError(
            "At least one path is required."
        )

    arguments = ["restore"]

    if staged:
        arguments.append("--staged")

    arguments.extend(
        [
            "--",
            *paths,
        ]
    )

    result = _run_git(
        workspace,
        arguments,
        write=True,
    )

    _require_success(
        result,
        "restore",
    )

    return {
        "operation": "restore",
        "paths": paths,
        "staged": staged,
        "success": True,
        "output": result.stdout.strip(),
    }