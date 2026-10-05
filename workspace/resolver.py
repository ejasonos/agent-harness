"""
Workspace path resolution utilities.
"""

from pathlib import Path


def normalize_path(path: str | Path) -> Path:
    """
    Expand environment variables, user home references,
    and resolve the path.
    """

    path = Path(path)

    return path.expanduser().resolve()


def is_relative_to(
    path: Path,
    parent: Path,
) -> bool:
    """
    Check whether path is contained inside parent.
    """

    try:
        path.resolve().relative_to(parent.resolve())

        return True

    except ValueError:
        return False