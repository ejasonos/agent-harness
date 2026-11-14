from pathlib import Path

import pytest

from tools.filesystem.read_many import read_files
from workspace.manager import WorkspaceManager


def test_read_files_returns_contents_and_sizes_in_input_order(tmp_path: Path):
    (tmp_path / "first.txt").write_text("one", encoding="utf-8")
    (tmp_path / "second.txt").write_text("two!", encoding="utf-8")
    workspace = WorkspaceManager(tmp_path)

    result = read_files(workspace, ["first.txt", "second.txt"])

    assert result["file_count"] == 2
    assert result["total_bytes"] == 7
    assert [file["path"] for file in result["files"]] == [
        "first.txt",
        "second.txt",
    ]
    assert [file["content"] for file in result["files"]] == ["one", "two!"]


def test_read_files_rejects_total_size_over_limit(tmp_path: Path):
    (tmp_path / "first.txt").write_text("1234", encoding="utf-8")
    (tmp_path / "second.txt").write_text("5678", encoding="utf-8")
    workspace = WorkspaceManager(tmp_path)

    with pytest.raises(ValueError, match="total limit"):
        read_files(
            workspace,
            ["first.txt", "second.txt"],
            max_total_bytes=7,
        )


def test_read_files_rejects_more_than_twenty_paths(tmp_path: Path):
    workspace = WorkspaceManager(tmp_path)

    with pytest.raises(ValueError, match="At most 20 files"):
        read_files(workspace, [f"{index}.txt" for index in range(21)])