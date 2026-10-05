from pathlib import Path

from workspace.manager import WorkspaceManager
from tools.filesystem.read import read_file
from tools.filesystem.write import write_file
from tools.filesystem.edit import edit_file
from tools.filesystem.search import search_files
from tools.filesystem.tree import directory_tree


workspace = WorkspaceManager(".")


test_file = Path("filesystem_integration_test.txt")


try:
    # WRITE
    print("\n=== WRITE ===")

    write_file(
        workspace,
        str(test_file),
        "Hello agent\n"
        "This is a test file\n"
        "Hello agent again\n",
    )

    print(read_file(workspace, str(test_file)))


    # EDIT
    print("\n=== EDIT ===")

    print(
        edit_file(
            workspace,
            str(test_file),
            "This is a test file",
            "This file was edited by the agent",
        )
    )

    print(read_file(workspace, str(test_file)))


    # SEARCH
    print("\n=== SEARCH ===")

    results = search_files(
        workspace,
        "agent",
        ".",
        extensions=[".txt"],
    )

    for result in results:
        print(result)


    # TREE
    print("\n=== TREE ===")

    print(
        directory_tree(
            workspace,
            ".",
            max_depth=3,
        )
    )


    # SECURITY
    print("\n=== SECURITY ===")

    try:
        read_file(
            workspace,
            "../outside.txt",
        )
    except PermissionError as exc:
        print("READ BLOCKED:", exc)


    try:
        edit_file(
            workspace,
            "../outside.txt",
            "old",
            "new",
        )
    except PermissionError as exc:
        print("EDIT BLOCKED:", exc)


finally:
    if test_file.exists():
        test_file.unlink()