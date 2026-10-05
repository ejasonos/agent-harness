import pytest

from tools.filesystem.delete import delete_file
from workspace.manager import WorkspaceManager


def test_delete_file_removes_file(tmp_path):
    workspace = WorkspaceManager(tmp_path, allow_delete=True)
    target = tmp_path / "example.txt"
    target.write_text("content", encoding="utf-8")

    result = delete_file(workspace, "example.txt")

    assert not target.exists()
    assert result == "Successfully deleted file: example.txt"


def test_delete_file_removes_directory_recursively(tmp_path):
    workspace = WorkspaceManager(tmp_path, allow_delete=True)
    target = tmp_path / "nested"
    target.mkdir()
    (target / "example.txt").write_text("content", encoding="utf-8")

    result = delete_file(workspace, "nested")

    assert not target.exists()
    assert result == "Successfully deleted directory: nested"


def test_delete_file_respects_delete_permission(tmp_path):
    workspace = WorkspaceManager(tmp_path)
    target = tmp_path / "example.txt"
    target.write_text("content", encoding="utf-8")

    with pytest.raises(PermissionError, match="Delete operations are disabled"):
        delete_file(workspace, "example.txt")

    assert target.exists()


def test_delete_file_rejects_workspace_root_and_outside_paths(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = WorkspaceManager(workspace_root, allow_delete=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("content", encoding="utf-8")

    with pytest.raises(PermissionError, match="workspace root"):
        delete_file(workspace, ".")

    with pytest.raises(PermissionError, match="outside workspace"):
        delete_file(workspace, "../outside.txt")

    assert outside.exists()


def test_delete_file_rejects_symlink_to_outside_workspace(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = WorkspaceManager(workspace_root, allow_delete=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("content", encoding="utf-8")
    link = workspace_root / "outside-link"

    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Unable to create symlink in this environment: {exc}")

    with pytest.raises(PermissionError):
        delete_file(workspace, "outside-link")

    assert outside.exists()