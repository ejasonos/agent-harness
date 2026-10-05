from pathlib import Path

from workspace.manager import WorkspaceManager
from tools.filesystem.write import write_file


workspace = WorkspaceManager(".")


test_path = "test_output.txt"

result = write_file(
    workspace,
    test_path,
    "Hello from the local agent!\nLine 2.",
)

print(result)


print("\nVERIFYING FILE:")

file_path = Path(test_path)

print("Exists:", file_path.exists())

if file_path.exists():
    print("Content:")
    print(file_path.read_text(encoding="utf-8"))


print("\nTESTING SECURITY:")

try:
    write_file(
        workspace,
        "../outside.txt",
        "This should never be written.",
    )
except PermissionError as exc:
    print("Blocked:", exc)


# Clean up the test file directly.
if file_path.exists():
    file_path.unlink()