from __future__ import annotations

import re
from typing import Any


DIAGNOSTIC_PATTERNS = [
    re.compile(
        r"^(?P<file>.+?):(?P<line>\d+):(?P<column>\d+):\s*"
        r"(?P<severity>error|warning|info)[:\s]+(?P<message>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<file>.+?)\((?P<line>\d+),(?P<column>\d+)\):\s*"
        r"(?P<severity>error|warning|info)\s*(?P<code>[A-Za-z0-9_-]+)?[:\s]*"
        r"(?P<message>.+)$",
        re.IGNORECASE,
    ),
]


def parse_diagnostics(output: str) -> list[dict[str, Any]]:
    """Parse common compiler/linter diagnostic formats."""

    diagnostics: list[dict[str, Any]] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        matched = None

        for pattern in DIAGNOSTIC_PATTERNS:
            matched = pattern.match(line)

            if matched:
                break

        if not matched:
            continue

        data = matched.groupdict()

        diagnostics.append(
            {
                "file": data.get("file"),
                "line": int(data["line"]) if data.get("line") else None,
                "column": (
                    int(data["column"])
                    if data.get("column")
                    else None
                ),
                "severity": (
                    data.get("severity", "error").lower()
                ),
                "code": data.get("code"),
                "message": data.get("message", "").strip(),
            }
        )

    return diagnostics


def collect_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    """Attach parsed diagnostics to a verification result."""

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")

    combined = "\n".join(
        value
        for value in (stdout, stderr)
        if value
    )

    diagnostics = parse_diagnostics(combined)

    enriched = dict(result)

    enriched["diagnostics"] = diagnostics
    enriched["diagnostic_count"] = len(diagnostics)

    enriched["errors"] = sum(
        1
        for item in diagnostics
        if item["severity"] == "error"
    )

    enriched["warnings"] = sum(
        1
        for item in diagnostics
        if item["severity"] == "warning"
    )

    return enriched