from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workspace.manager import WorkspaceManager
from tools.testing.runner import _quote, run_process


def _detect_package_manager(root: Path) -> str:
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm"

    if (root / "yarn.lock").exists():
        return "yarn"

    if (root / "bun.lockb").exists() or (root / "bun.lock").exists():
        return "bun"

    return "npm"


def _node_test_command(root: Path) -> str:
    package_json = root / "package.json"

    if not package_json.exists():
        return "npm test"

    try:
        package = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "npm test"

    scripts = package.get("scripts", {})

    if "test" not in scripts:
        return "npm test"

    manager = _detect_package_manager(root)

    if manager == "npm":
        return "npm test"

    return f"{manager} test"


def detect_test_command(workspace: WorkspaceManager) -> str:
    """Determine a reasonable default test command."""

    root = workspace.root

    if (
        (root / "pyproject.toml").exists()
        or (root / "pytest.ini").exists()
        or (root / "tests").exists()
        or (root / "test").exists()
    ):
        return "pytest"

    if (root / "package.json").exists():
        return _node_test_command(root)

    if (root / "go.mod").exists():
        return "go test ./..."

    if (root / "Cargo.toml").exists():
        return "cargo test"

    if (root / "pom.xml").exists():
        return "mvn test"

    if (root / "gradlew").exists():
        return "./gradlew test"

    if (root / "gradlew.bat").exists():
        return "gradlew.bat test"

    raise RuntimeError(
        "Could not automatically determine a test command. "
        "Provide command explicitly."
    )


def run_tests(
    workspace: WorkspaceManager,
    command: str | None = None,
    *,
    cwd: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Run the project's test suite."""

    resolved_command = command or detect_test_command(workspace)

    result = run_process(
        workspace,
        resolved_command,
        cwd=cwd,
        timeout=timeout,
    )

    result["operation"] = "test"

    return result


def run_test_file(
    workspace: WorkspaceManager,
    path: str,
    *,
    command: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Run tests associated with a specific test file."""

    file_path = workspace.read_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    if not file_path.is_file():
        raise IsADirectoryError(f"Test path is not a file: {path}")

    suffix = file_path.suffix.lower()

    if command:
        resolved_command = f"{command} {_quote(str(file_path))}"
    elif suffix == ".py":
        resolved_command = f"pytest {_quote(str(file_path))}"
    elif suffix in {".js", ".jsx", ".ts", ".tsx"}:
        resolved_command = f"npm test -- {_quote(str(file_path))}"
    elif suffix == ".go":
        resolved_command = f"go test {_quote(str(file_path))}"
    elif suffix == ".rs":
        resolved_command = "cargo test"
    else:
        raise RuntimeError(
            f"Cannot automatically determine how to run test file: {path}"
        )

    result = run_process(
        workspace,
        resolved_command,
        timeout=timeout,
    )

    result["operation"] = "test_file"
    result["test_file"] = path

    return result