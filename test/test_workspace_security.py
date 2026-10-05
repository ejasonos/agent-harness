from workspace.security import WorkspaceSecurity


security = WorkspaceSecurity(
    ".",
    allow_write=True,
    allow_delete=False,
)


print("Workspace:", security.workspace)


# Should succeed
print(
    "Inside:",
    security.resolve("test.txt"),
)


# Should fail
try:
    security.resolve("../outside.txt")
except PermissionError as exc:
    print("Blocked:", exc)


# Delete should be blocked
try:
    security.check_delete("test.txt")
except PermissionError as exc:
    print("Delete blocked:", exc)