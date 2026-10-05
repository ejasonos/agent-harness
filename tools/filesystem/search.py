from __future__ import annotations

from pathlib import Path

from workspace.manager import WorkspaceManager


def search_files(
    workspace: WorkspaceManager,
    query: str,
    path: str = ".",
    *,
    extensions: list[str] | None = None,
    max_results: int = 100,
) -> list[dict[str, object]]:
    """
    Search text files inside the workspace.

    Returns matching file paths, line numbers and lines.
    """

    root = workspace.resolve(path)

    if not root.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if root.is_file():
        files = [root]
    else:
        files = [
            item
            for item in root.rglob("*")
            if item.is_file()
        ]

    normalized_extensions = None

    if extensions:
        normalized_extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in extensions
        }

    results: list[dict[str, object]] = []

    for file_path in files:
        if normalized_extensions:
            if file_path.suffix.lower() not in normalized_extensions:
                continue

        try:
            content = file_path.read_text(
                encoding="utf-8"
            )
        except (UnicodeDecodeError, PermissionError):
            continue

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):
            if query.lower() in line.lower():
                results.append(
                    {
                        "path": str(
                            file_path.relative_to(
                                workspace.root
                            )
                        ),
                        "line": line_number,
                        "content": line,
                    }
                )

                if len(results) >= max_results:
                    return results

    return results