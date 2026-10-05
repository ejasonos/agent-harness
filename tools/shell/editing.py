from __future__ import annotations

import shlex


def quote_argument(
    value: str,
) -> str:
    """
    Safely quote one shell argument for the current platform.
    """

    if not isinstance(value, str):
        raise TypeError(
            "value must be a string."
        )

    if not value:
        return '""'

    return shlex.quote(value)


def build_command(
    executable: str,
    arguments: list[str] | None = None,
) -> str:
    """
    Build a shell command from an executable and arguments.

    Arguments are quoted individually rather than concatenated
    blindly.
    """

    executable = executable.strip()

    if not executable:
        raise ValueError(
            "Executable cannot be empty."
        )

    parts = [
        quote_argument(executable)
    ]

    for argument in arguments or []:
        if not isinstance(argument, str):
            raise TypeError(
                "All arguments must be strings."
            )

        parts.append(
            quote_argument(argument)
        )

    return " ".join(parts)


def parse_command(
    command: str,
) -> list[str]:
    """
    Parse a command into shell-like arguments.

    This is useful for inspecting a command before execution.
    """

    command = command.strip()

    if not command:
        raise ValueError(
            "Command cannot be empty."
        )

    try:
        return shlex.split(command)
    except ValueError as exc:
        raise ValueError(
            f"Invalid shell command: {exc}"
        ) from exc


def inspect_command(
    command: str,
) -> dict[str, object]:
    """
    Parse a command and expose its executable and arguments
    without executing it.
    """

    parts = parse_command(command)

    return {
        "command": command,
        "executable": parts[0],
        "arguments": parts[1:],
        "argument_count": len(parts) - 1,
    }