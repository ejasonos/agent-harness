from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any

from workspace.manager import WorkspaceManager


def _quote(value: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


def run_process(
    workspace: WorkspaceManager,
    command: str,
    *,
    cwd: str | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a verification command and return structured results."""

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    workspace.check_shell()

    if timeout > 600:
        timeout = 600

    working_directory = workspace.root

    if cwd:
        working_directory = workspace.read_path(cwd)

        if not working_directory.exists():
            raise FileNotFoundError(f"Working directory not found: {cwd}")

        if not working_directory.is_dir():
            raise NotADirectoryError(f"Working directory is not a directory: {cwd}")

    process_env = os.environ.copy()

    if env:
        process_env.update(env)

    try:
        completed = subprocess.run(
            command,
            cwd=str(working_directory),
            env=process_env,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "command": command,
            "cwd": str(working_directory),
            "success": completed.returncode == 0,
            "timed_out": False,
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(working_directory),
            "success": False,
            "timed_out": True,
            "return_code": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }