from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.git.operations import git_add
from tools.shell.execute import execute_command
from tools.testing.runner import run_process
from tools.testing.tests import detect_test_command, run_test_file
from workspace.manager import WorkspaceManager


def _assert_permission_denied(operation):
    try:
        operation()
    except PermissionError:
        return
    raise AssertionError("expected operation to be denied")


def test_disabled_execution_permissions_are_enforced():
    with TemporaryDirectory() as directory:
        workspace = WorkspaceManager(
            directory,
            allow_shell=False,
            allow_network=False,
            allow_git_write=False,
        )

        _assert_permission_denied(
            lambda: execute_command(workspace, "echo blocked")
        )
        _assert_permission_denied(
            lambda: run_process(workspace, "echo blocked")
        )
        _assert_permission_denied(
            lambda: git_add(workspace, ["README.md"])
        )
        _assert_permission_denied(workspace.check_network)


def test_python_test_command_wins_over_node_manifest():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "pytest.ini").write_text(
            "[pytest]\ntestpaths = tests\n",
            encoding="utf-8",
        )
        (root / "package.json").write_text(
            '{"scripts":{"test":"echo no tests && exit 1"}}',
            encoding="utf-8",
        )

        assert detect_test_command(
            WorkspaceManager(root)
        ) == "pytest"


def test_run_test_file_builds_a_quoted_command():
    with TemporaryDirectory() as directory:
        workspace = WorkspaceManager(directory)
        test_file = workspace.root / "test sample.py"
        test_file.write_text("", encoding="utf-8")

        with patch(
            "tools.testing.tests.run_process",
            return_value={"success": True},
        ) as run:
            run_test_file(workspace, "test sample.py")

        command = run.call_args.args[1]
        assert command.startswith("pytest ")
        assert "test sample.py" in command