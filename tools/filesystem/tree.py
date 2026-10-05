from __future__ import annotations

from workspace.manager import WorkspaceManager


def directory_tree(
    workspace: WorkspaceManager,
    path: str = ".",
    *,
    max_depth: int = 4,
) -> str:
    """
    Return a readable directory tree.
    """

    root = workspace.resolve(path)

    if not root.exists():
        raise FileNotFoundError(
            f"Path not found: {path}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {path}"
        )

    lines: list[str] = [root.name]

    def walk(directory, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return

        entries = sorted(
            directory.iterdir(),
            key=lambda item: (
                item.is_file(),
                item.name.lower(),
            ),
        )

        for index, entry in enumerate(entries):
            is_last = index == len(entries) - 1
            connector = "└── " if is_last else "├── "

            lines.append(
                f"{prefix}{connector}{entry.name}"
            )

            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk(
                    entry,
                    prefix + extension,
                    depth + 1,
                )

    walk(root, "", 0)

    return "\n".join(lines)