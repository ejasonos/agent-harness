from __future__ import annotations

from workspace.manager import WorkspaceManager


def write_file(
    workspace: WorkspaceManager,
    path: str,
    content: str,
) -> str:
    """
    Write text content to a file inside the agent workspace.
    """

    file_path = workspace.write_path(path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    return (
        f"Successfully wrote {len(content)} characters "
        f"to {path}"
    )