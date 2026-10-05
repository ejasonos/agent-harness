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


def git_history(
    workspace: WorkspaceManager,
    limit: int = 20,
    path: str | None = None,
) -> list[dict[str, str]]:
    """
    Return recent Git commits.

    The result is normalized into commit records rather than
    returning raw git log output.
    """

    if limit < 1:
        raise ValueError(
            "limit must be >= 1"
        )

    if limit > 200:
        raise ValueError(
            "limit must be <= 200"
        )

    format_string = (
        "%H%x1f%h%x1f%an%x1f%ae%x1f%aI%x1f%s%x1e"
    )

    arguments = [
        "log",
        f"-n{limit}",
        f"--format={format_string}",
    ]

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
        "log",
    )

    commits: list[dict[str, str]] = []

    for record in result.stdout.split("\x1e"):
        record = record.strip()

        if not record:
            continue

        fields = record.split("\x1f")

        if len(fields) != 6:
            continue

        (
            commit_hash,
            short_hash,
            author,
            email,
            authored_at,
            subject,
        ) = fields

        commits.append(
            {
                "hash": commit_hash,
                "short_hash": short_hash,
                "author": author,
                "email": email,
                "date": authored_at,
                "subject": subject,
            }
        )

    return commits