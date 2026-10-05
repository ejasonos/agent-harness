from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping

from workspace.manager import WorkspaceManager


DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600


def _resolve_working_directory(
    workspace: WorkspaceManager,
    cwd: str | None,
) -> Path:
    if cwd is None:
        return Path(workspace.root)

    path = workspace.read_path(cwd)

    if not path.exists():
        raise FileNotFoundError(
            f"Working directory not found: {cwd}"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"Working directory is not a directory: {cwd}"
        )

    return path


def _normalize_environment(
    env: Mapping[str, str] | None,
) -> dict[str, str]:
    environment = dict(os.environ)

    if env is not None:
        for key, value in env.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Environment variable names must be strings."
                )

            if not isinstance(value, str):
                raise TypeError(
                    "Environment variable values must be strings."
                )

            environment[key] = value

    return environment


def execute_command(
    workspace: WorkspaceManager,
    command: str,
    cwd: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """
    Execute a shell command inside the agent workspace.

    The command is executed through the platform shell so that
    normal project commands such as npm, python, git, and scripts
    work naturally.
    """

    command = command.strip()

    if not command:
        raise ValueError(
            "Command cannot be empty."
        )

    workspace.check_shell()

    if timeout < 1:
        raise ValueError(
            "timeout must be >= 1 second."
        )

    if timeout > MAX_TIMEOUT:
        raise ValueError(
            f"timeout must be <= {MAX_TIMEOUT} seconds."
        )

    working_directory = _resolve_working_directory(
        workspace,
        cwd,
    )

    environment = _normalize_environment(env)

    try:
        result = subprocess.run(
            command,
            cwd=working_directory,
            env=environment,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(
                "utf-8",
                errors="replace",
            )
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )

        stderr = (
            exc.stderr.decode(
                "utf-8",
                errors="replace",
            )
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )

        return {
            "command": command,
            "cwd": str(working_directory),
            "success": False,
            "timed_out": True,
            "return_code": None,
            "stdout": stdout,
            "stderr": stderr,
            "error": (
                f"Command timed out after {timeout} seconds."
            ),
        }
    except OSError as exc:
        raise RuntimeError(
            f"Failed to execute command: {exc}"
        ) from exc

    return {
        "command": command,
        "cwd": str(working_directory),
        "success": result.returncode == 0,
        "timed_out": False,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }