from __future__ import annotations

from workspace.manager import WorkspaceManager

def read_file(
    workspace: WorkspaceManager,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    max_bytes: int | None = None,
) -> str:
    """
    Read a text file inside the agent workspace.

    Lines are 1-based and inclusive.
    max_bytes limits the raw file size when supplied by a batch reader.
    """

    if max_bytes is not None and max_bytes < 1:
        raise ValueError("max_bytes must be >= 1.")

    file_path = workspace.read_path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if not file_path.is_file():
        raise IsADirectoryError(
            f"Path is not a file: {path}"
        )

    if max_bytes is None:
        content = file_path.read_text(encoding="utf-8")
    else:
        with file_path.open("rb") as file:
            data = file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(
                f"File {path!r} exceeds the {max_bytes}-byte read limit."
            )
        content = data.decode("utf-8")

    lines = content.splitlines()

    if start_line is None and end_line is None:
        return content

    start = 1 if start_line is None else start_line
    end = len(lines) if end_line is None else end_line

    if start < 1:
        raise ValueError(
            "start_line must be >= 1"
        )

    if end < start:
        raise ValueError(
            "end_line must be >= start_line"
        )

    selected_lines = lines[start - 1:end]

    return "\n".join(selected_lines)