from workspace.manager import WorkspaceManager


workspace = WorkspaceManager(
    ".",
    allow_delete=False,
)


print("Workspace root:")
print(workspace.root)


print("\nValid path:")
print(workspace.resolve("src/test.py"))


print("\nTesting blocked path:")

try:
    workspace.resolve("../outside.txt")
except PermissionError as exc:
    print("Blocked:", exc)


print("\nTesting blocked delete:")

try:
    workspace.delete_path("test.py")
except PermissionError as exc:
    print("Blocked:", exc)