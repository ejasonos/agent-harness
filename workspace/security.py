from __future__ import annotations

from pathlib import Path


class WorkspaceSecurity:
    """
    Enforces the agent's workspace boundary.

    The agent may only access files and directories inside
    the configured workspace.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        allow_delete: bool = False,
        allow_write: bool = True,
        allow_git_write: bool = False,
        allow_shell: bool = True,
        allow_network: bool = True,
    ) -> None:
        self.workspace = Path(workspace).resolve()

        self.allow_delete = allow_delete
        self.allow_write = allow_write
        self.allow_git_write = allow_git_write
        self.allow_shell = allow_shell
        self.allow_network = allow_network

    def resolve(self, path: str | Path) -> Path:
        """
        Resolve a path and ensure it remains inside the workspace.
        """

        candidate = Path(path)

        if not candidate.is_absolute():
            candidate = self.workspace / candidate

        resolved = candidate.resolve()

        if not self.is_inside(resolved):
            raise PermissionError(
                f"Access denied: path is outside workspace: {path}"
            )

        return resolved

    def is_inside(self, path: str | Path) -> bool:
        """
        Return True if the path is inside the workspace.
        """

        try:
            Path(path).resolve().relative_to(self.workspace)
            return True
        except ValueError:
            return False

    def check_read(self, path: str | Path) -> Path:
        """
        Validate a read operation.
        """

        return self.resolve(path)

    def check_write(self, path: str | Path) -> Path:
        """
        Validate a write operation.
        """

        if not self.allow_write:
            raise PermissionError(
                "Write operations are disabled."
            )

        return self.resolve(path)

    def check_delete(self, path: str | Path) -> Path:
        """
        Validate a delete operation.
        """

        if not self.allow_delete:
            raise PermissionError(
                "Delete operations are disabled."
            )

        return self.resolve(path)

    def check_shell(self) -> None:
        """
        Validate shell execution.
        """

        if not self.allow_shell:
            raise PermissionError(
                "Shell execution is disabled."
            )

    def check_git_write(self) -> None:
        if not self.allow_git_write:
            raise PermissionError(
                "Git write operations are disabled."
            )

    def check_network(self) -> None:
        """
        Validate network access.
        """

        if not self.allow_network:
            raise PermissionError(
                "Network access is disabled."
            )