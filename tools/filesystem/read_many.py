from __future__ import annotations

from typing import Any

from config.defaults import MAX_FILE_READ_BYTES
from tools.filesystem.read import read_file
from workspace.manager import WorkspaceManager


MAX_READ_FILES = 20
DEFAULT_MAX_TOTAL_READ_BYTES = 4 * MAX_FILE_READ_BYTES


def read_files(
    workspace: WorkspaceManager,
    paths: list[str],
    max_bytes_per_file: int = MAX_FILE_READ_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_READ_BYTES,
) -> dict[str, Any]:
    """Read several workspace files with explicit per-file and total limits."""
    if not paths:
        raise ValueError("paths must contain at least one file.")
    if len(paths) > MAX_READ_FILES:
        raise ValueError(f"At most {MAX_READ_FILES} files can be read at once.")
    if max_bytes_per_file < 1:
        raise ValueError("max_bytes_per_file must be >= 1.")
    if max_total_bytes < 1:
        raise ValueError("max_total_bytes must be >= 1.")

    file_sizes: list[tuple[str, int]] = []
    total_bytes = 0
    for path in paths:
        file_path = workspace.read_path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {path}")

        size = file_path.stat().st_size
        if size > max_bytes_per_file:
            raise ValueError(
                f"File {path!r} is {size} bytes, above the "
                f"{max_bytes_per_file}-byte per-file limit."
            )
        total_bytes += size
        if total_bytes > max_total_bytes:
            raise ValueError(
                f"Files total {total_bytes} bytes, above the "
                f"{max_total_bytes}-byte total limit."
            )
        file_sizes.append((path, size))

    files = []
    actual_total_bytes = 0
    for path, _ in file_sizes:
        remaining_bytes = max_total_bytes - actual_total_bytes
        if remaining_bytes < 1:
            raise ValueError(
                f"Files exceed the {max_total_bytes}-byte total read limit."
            )

        content = read_file(
            workspace,
            path,
            max_bytes=min(max_bytes_per_file, remaining_bytes),
        )
        size = len(content.encode("utf-8"))
        actual_total_bytes += size
        files.append(
            {
                "path": path,
                "size_bytes": size,
                "content": content,
            }
        )

    return {
        "files": files,
        "file_count": len(files),
        "total_bytes": actual_total_bytes,
    }
