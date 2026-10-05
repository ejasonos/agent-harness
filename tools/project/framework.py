from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workspace.manager import WorkspaceManager


NODE_FRAMEWORKS: dict[str, str] = {
    "next": "Next.js",
    "react": "React",
    "react-dom": "React",
    "vue": "Vue",
    "nuxt": "Nuxt",
    "@angular/core": "Angular",
    "svelte": "Svelte",
    "@sveltejs/kit": "SvelteKit",
    "express": "Express",
    "fastify": "Fastify",
    "hono": "Hono",
    "nestjs": "NestJS",
    "@nestjs/core": "NestJS",
    "vite": "Vite",
    "electron": "Electron",
    "expo": "Expo",
    "react-native": "React Native",
}

PYTHON_FRAMEWORKS: dict[str, str] = {
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "starlette": "Starlette",
    "tornado": "Tornado",
    "gradio": "Gradio",
}


def _load_package_json(
    project_path: Path,
) -> dict[str, Any]:
    package_json = project_path / "package.json"

    if not package_json.exists():
        return {}

    try:
        data = json.loads(
            package_json.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def _detect_node_frameworks(
    project_path: Path,
) -> list[str]:
    package = _load_package_json(project_path)

    dependencies: dict[str, Any] = {}

    for section in (
        "dependencies",
        "devDependencies",
        "peerDependencies",
        "optionalDependencies",
    ):
        values = package.get(section, {})

        if isinstance(values, dict):
            dependencies.update(values)

    frameworks: list[str] = []

    for dependency, framework in NODE_FRAMEWORKS.items():
        if dependency in dependencies and framework not in frameworks:
            frameworks.append(framework)

    return frameworks


def _detect_python_frameworks(
    project_path: Path,
) -> list[str]:
    candidates = [
        project_path / "pyproject.toml",
        project_path / "requirements.txt",
        project_path / "requirements-dev.txt",
        project_path / "Pipfile",
        project_path / "setup.py",
    ]

    contents: list[str] = []

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue

        try:
            contents.append(
                candidate.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).lower()
            )
        except OSError:
            continue

    combined = "\n".join(contents)

    frameworks: list[str] = []

    for dependency, framework in PYTHON_FRAMEWORKS.items():
        if dependency in combined:
            frameworks.append(framework)

    return frameworks


def detect_framework(
    workspace: WorkspaceManager,
    path: str = ".",
) -> dict[str, object]:
    """
    Detect application frameworks and major development technologies
    from project configuration files.
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

    frameworks: list[str] = []

    frameworks.extend(
        _detect_node_frameworks(project_path)
    )

    frameworks.extend(
        _detect_python_frameworks(project_path)
    )

    if (project_path / "go.mod").exists():
        frameworks.append("Go")

    if (project_path / "Cargo.toml").exists():
        frameworks.append("Rust")

    if (project_path / "composer.json").exists():
        frameworks.append("Composer/PHP")

    if (project_path / "Gemfile").exists():
        frameworks.append("Ruby")

    # Remove duplicates while preserving discovery order.
    frameworks = list(dict.fromkeys(frameworks))

    return {
        "path": str(project_path),
        "frameworks": frameworks,
        "framework_count": len(frameworks),
    }