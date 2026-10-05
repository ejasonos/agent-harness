from workspace.manager import WorkspaceManager
from tools.filesystem.read import read_file


workspace = WorkspaceManager(".")


content = read_file(
    workspace,
    "test_filesystem_read.py",
)

print("FULL FILE:")
print(content)


print("\nSELECTED LINES:")

selected = read_file(
    workspace,
    "test_filesystem_read.py",
    start_line=1,
    end_line=5,
)

print(selected)


print("\nTESTING SECURITY:")

try:
    read_file(
        workspace,
        "../outside.txt",
    )
except PermissionError as exc:
    print("Blocked:", exc)