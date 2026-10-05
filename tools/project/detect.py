from __future__ import annotations

from pathlib import Path

from workspace.manager import WorkspaceManager


PROJECT_MARKERS: dict[str, str] = {
    "package.json": "node",
    "pnpm-lock.yaml": "node",
    "yarn.lock": "node",
    "package-lock.json": "node",
    "bun.lock": "node",
    "bun.lockb": "node",
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "Pipfile": "python",
    "poetry.lock": "python",
    "setup.py": "python",
    "go.mod": "go",
    "Cargo.toml": "rust",
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "java",
    "composer.json": "php",
    "Gemfile": "ruby",
    "mix.exs": "elixir",
    "pubspec.yaml": "dart",
}


def detect_project(
    workspace: WorkspaceManager,
    path: str = ".",
) -> dict[str, object]:
    """
    Detect the primary project type from files in the workspace.

    The result contains the detected project type, project root,
    marker files, and useful metadata for downstream tools.
    """

    project_path = workspace.read_path(path)

    if not project_path.exists():
        raise FileNotFoundError(
            f"Project path not found: {path}"
        )

    if not project_path.is_dir():
        raise NotADirectoryError(
            f"Project path is not a directory: {path}"
        )

    marker_files: list[str] = []
    detected_types: list[str] = []

    for marker, project_type in PROJECT_MARKERS.items():
        marker_path = project_path / marker

        if marker_path.exists() and marker_path.is_file():
            marker_files.append(marker)

            if project_type not in detected_types:
                detected_types.append(project_type)

    project_type: str

    if not detected_types:
        project_type = "unknown"
    elif len(detected_types) == 1:
        project_type = detected_types[0]
    else:
        project_type = "multi"

    return {
        "path": str(project_path),
        "project_type": project_type,
        "detected_types": detected_types,
        "marker_files": marker_files,
        "is_project": bool(marker_files),
    }