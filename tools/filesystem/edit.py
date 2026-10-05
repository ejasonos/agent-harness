from __future__ import annotations

from workspace.manager import WorkspaceManager


def edit_file(
    workspace: WorkspaceManager,
    path: str,
    old_text: str,
    new_text: str,
    *,
    replace_all: bool = False,
) -> str:
    """
    Replace text inside a workspace file.

    By default, exactly one occurrence must exist.
    """

    file_path = workspace.write_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")

    content = file_path.read_text(encoding="utf-8")

    occurrences = content.count(old_text)

    if occurrences == 0:
        raise ValueError(
            f"Target text was not found in {path}"
        )

    if not replace_all and occurrences > 1:
        raise ValueError(
            f"Target text occurs {occurrences} times. "
            "Use replace_all=True to replace every occurrence."
        )

    if replace_all:
        updated = content.replace(old_text, new_text)
        replacements = occurrences
    else:
        updated = content.replace(old_text, new_text, 1)
        replacements = 1

    file_path.write_text(
        updated,
        encoding="utf-8",
    )

    return (
        f"Successfully edited {path}. "
        f"Replacements: {replacements}"
    )