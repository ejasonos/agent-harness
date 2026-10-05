from __future__ import annotations

from pathlib import Path

from workspace.security import WorkspaceSecurity


class WorkspaceManager:
    """
    Central manager for the agent's workspace.

    All filesystem paths should pass through this class so that
    workspace security is enforced consistently.
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
        self.security = WorkspaceSecurity(
            workspace,
            allow_delete=allow_delete,
            allow_write=allow_write,
            allow_git_write=allow_git_write,
            allow_shell=allow_shell,
            allow_network=allow_network,
        )

    @property
    def root(self) -> Path:
        return self.security.workspace

    def resolve(self, path: str | Path) -> Path:
        return self.security.resolve(path)

    def read_path(self, path: str | Path) -> Path:
        return self.security.check_read(path)

    def write_path(self, path: str | Path) -> Path:
        return self.security.check_write(path)

    def delete_path(self, path: str | Path) -> Path:
        return self.security.check_delete(path)

    def check_shell(self) -> None:
        self.security.check_shell()

    def check_git_write(self) -> None:
        self.security.check_git_write()

    def check_network(self) -> None:
        self.security.check_network()