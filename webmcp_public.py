from __future__ import annotations

import re
from pathlib import Path

from fastmcp import FastMCP


mcp = FastMCP("Local Agent Public Documentation")

_README_PATH = Path(__file__).resolve().with_name("README.md")
_MAX_README_BYTES = 200_000
_PUBLIC_SECTIONS = {
    "overview": ("Local Agent", True),
    "available-tool-groups": ("Available Tool Groups", False),
    "workspace-safety": ("Workspace Safety", False),
}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _read_public_section(section: str) -> dict[str, str]:
    if section not in _PUBLIC_SECTIONS:
        raise ValueError(
            "section must be one of: "
            + ", ".join(sorted(_PUBLIC_SECTIONS))
        )

    size = _README_PATH.stat().st_size
    if size > _MAX_README_BYTES:
        raise RuntimeError("README.md exceeds the public documentation size limit.")

    lines = _README_PATH.read_text(encoding="utf-8").splitlines()
    heading, stop_at_any_heading = _PUBLIC_SECTIONS[section]

    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if not match or match.group(2).strip().casefold() != heading.casefold():
            continue

        heading_level = len(match.group(1))
        content: list[str] = []
        for following_line in lines[index + 1:]:
            next_heading = _HEADING_RE.match(following_line)
            if next_heading and (
                stop_at_any_heading
                or len(next_heading.group(1)) <= heading_level
            ):
                break
            content.append(following_line)

        return {
            "section": section,
            "title": heading,
            "text": "\n".join(content).strip()[:8_000],
        }

    raise RuntimeError(f"Public README section is missing: {heading}")


@mcp.tool()
def get_project_overview() -> dict[str, str]:
    """Return a short overview of this repository from its public README."""
    return _read_public_section("overview")


@mcp.tool()
def search_project_docs(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search the public README sections about project tools and workspace safety."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")
    if len(normalized_query) > 160:
        raise ValueError("query must be at most 160 characters.")
    if limit < 1 or limit > 10:
        raise ValueError("limit must be between 1 and 10.")

    results: list[dict[str, str]] = []
    for section in ("available-tool-groups", "workspace-safety"):
        document = _read_public_section(section)
        in_code_block = False
        for line in document["text"].splitlines():
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            excerpt = line.strip()
            if re.search(
                rf"(?<!\w){re.escape(normalized_query)}(?!\w)",
                excerpt,
                re.IGNORECASE,
            ):
                results.append(
                    {
                        "section": section,
                        "title": document["title"],
                        "excerpt": excerpt[:500],
                    }
                )
                if len(results) >= limit:
                    return results

    return results


@mcp.tool()
def read_project_doc(section: str) -> dict[str, str]:
    """Read one allowlisted public README section."""
    normalized_section = re.sub(
        r"[^a-z0-9]+",
        "-",
        section.strip().casefold(),
    ).strip("-")
    return _read_public_section(normalized_section)
