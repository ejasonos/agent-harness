from __future__ import annotations

import os
import signal
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from workspace.manager import WorkspaceManager


@dataclass
class ManagedProcess:
    process_id: str
    command: str
    cwd: str
    process: subprocess.Popen[str]
    stdout_lines: list[str] = field(
        default_factory=list
    )
    stderr_lines: list[str] = field(
        default_factory=list
    )
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None


class ProcessManager:
    """
    Maintains long-running shell processes for the agent.

    Processes are identified by stable agent-generated IDs rather
    than exposing operating-system PIDs as the primary interface.
    """

    def __init__(self) -> None:
        self._processes: dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()

    def _reader(
        self,
        stream,
        target: list[str],
    ) -> None:
        try:
            for line in iter(stream.readline, ""):
                target.append(line)
        finally:
            stream.close()

    def start(
        self,
        workspace: WorkspaceManager,
        command: str,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        command = command.strip()

        if not command:
            raise ValueError(
                "Command cannot be empty."
            )

        workspace.check_shell()

        if cwd is None:
            working_directory = Path(
                workspace.root
            )
        else:
            working_directory = workspace.read_path(
                cwd
            )

        if not working_directory.exists():
            raise FileNotFoundError(
                f"Working directory not found: {cwd}"
            )

        if not working_directory.is_dir():
            raise NotADirectoryError(
                f"Working directory is not a directory: {cwd}"
            )

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

        creation_flags = 0

        if os.name == "nt":
            creation_flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
            )

        try:
            process = subprocess.Popen(
                command,
                cwd=working_directory,
                env=environment,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Failed to start process: {exc}"
            ) from exc

        process_id = uuid.uuid4().hex

        managed = ManagedProcess(
            process_id=process_id,
            command=command,
            cwd=str(working_directory),
            process=process,
        )

        stdout_thread = threading.Thread(
            target=self._reader,
            args=(
                process.stdout,
                managed.stdout_lines,
            ),
            daemon=True,
        )

        stderr_thread = threading.Thread(
            target=self._reader,
            args=(
                process.stderr,
                managed.stderr_lines,
            ),
            daemon=True,
        )

        managed.stdout_thread = stdout_thread
        managed.stderr_thread = stderr_thread

        with self._lock:
            self._processes[process_id] = managed

        stdout_thread.start()
        stderr_thread.start()

        return {
            "process_id": process_id,
            "pid": process.pid,
            "command": command,
            "cwd": str(working_directory),
            "running": process.poll() is None,
        }

    def _get(
        self,
        process_id: str,
    ) -> ManagedProcess:
        with self._lock:
            managed = self._processes.get(process_id)

        if managed is None:
            raise KeyError(
                f"Unknown process ID: {process_id}"
            )

        return managed

    def status(
        self,
        process_id: str,
    ) -> dict[str, object]:
        managed = self._get(process_id)

        return_code = managed.process.poll()

        return {
            "process_id": process_id,
            "pid": managed.process.pid,
            "command": managed.command,
            "cwd": managed.cwd,
            "running": return_code is None,
            "return_code": return_code,
            "stdout": "".join(
                managed.stdout_lines
            ),
            "stderr": "".join(
                managed.stderr_lines
            ),
        }

    def stop(
        self,
        process_id: str,
        force: bool = False,
    ) -> dict[str, object]:
        managed = self._get(process_id)

        process = managed.process

        if process.poll() is not None:
            return self.status(process_id)

        try:
            if os.name == "nt":
                if force:
                    process.kill()
                else:
                    process.send_signal(
                        signal.CTRL_BREAK_EVENT
                    )
            else:
                if force:
                    process.kill()
                else:
                    process.terminate()
        except OSError as exc:
            raise RuntimeError(
                f"Failed to stop process: {exc}"
            ) from exc

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

        return self.status(process_id)

    def remove(
        self,
        process_id: str,
    ) -> None:
        with self._lock:
            managed = self._processes.pop(
                process_id,
                None,
            )

        if managed is None:
            raise KeyError(
                f"Unknown process ID: {process_id}"
            )


_PROCESS_MANAGER = ProcessManager()


def start_process(
    workspace: WorkspaceManager,
    command: str,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Start a persistent background shell process."""

    return _PROCESS_MANAGER.start(
        workspace,
        command,
        cwd=cwd,
        env=env,
    )


def process_status(
    process_id: str,
) -> dict[str, object]:
    """Return the current state and captured output of a process."""

    return _PROCESS_MANAGER.status(
        process_id
    )


def stop_process(
    process_id: str,
    force: bool = False,
) -> dict[str, object]:
    """Stop a managed process."""

    return _PROCESS_MANAGER.stop(
        process_id,
        force=force,
    )


def remove_process(
    process_id: str,
) -> dict[str, object]:
    """Remove a completed process from the process registry."""

    _PROCESS_MANAGER.remove(process_id)

    return {
        "process_id": process_id,
        "removed": True,
    }