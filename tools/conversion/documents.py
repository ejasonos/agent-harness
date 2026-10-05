from __future__ import annotations

from pathlib import Path
from typing import Any

from workspace.manager import WorkspaceManager

from .engine import (
    resolve_input,
    resolve_output,
    require_dependency,
)


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".py",
    ".java",
    ".go",
    ".rs",
    ".sql",
}


def extract_document_text(
    workspace: WorkspaceManager,
    path: str,
) -> dict[str, Any]:

    source = resolve_input(
        workspace,
        path,
    )

    suffix = source.suffix.lower()

    if suffix in TEXT_EXTENSIONS:
        text = source.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return {
            "path": path,
            "format": suffix.lstrip("."),
            "text": text,
            "character_count": len(text),
        }

    if suffix == ".docx":
        return _extract_docx(
            source,
            path,
        )

    if suffix == ".pdf":
        from .pdf import extract_pdf_text

        return extract_pdf_text(
            workspace,
            path,
        )

    raise ValueError(
        f"Unsupported document format: {suffix}"
    )


def _extract_docx(
    source: Path,
    original_path: str,
) -> dict[str, Any]:

    docx = require_dependency(
        "docx",
        "python-docx",
    )

    document = docx.Document(
        str(source)
    )

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    text = "\n".join(paragraphs)

    return {
        "path": original_path,
        "format": "docx",
        "text": text,
        "character_count": len(text),
        "paragraph_count": len(paragraphs),
    }


def convert_document(
    workspace: WorkspaceManager,
    source_path: str,
    output_path: str,
    output_format: str,
) -> dict[str, Any]:

    source = resolve_input(
        workspace,
        source_path,
    )

    destination = resolve_output(
        workspace,
        output_path,
    )

    output_format = output_format.lower().lstrip(".")

    if output_format == "txt":
        return _convert_to_text(
            source,
            destination,
            source_path,
        )

    if output_format == "docx":
        return _convert_to_docx(
            source,
            destination,
            source_path,
        )

    if output_format == "pdf":
        return _convert_to_pdf(
            source,
            destination,
            source_path,
        )

    raise ValueError(
        f"Unsupported output format: {output_format}"
    )


def _convert_to_text(
    source: Path,
    destination: Path,
    source_path: str,
) -> dict[str, Any]:

    suffix = source.suffix.lower()

    if suffix == ".docx":
        result = _extract_docx(
            source,
            source_path,
        )

        text = result["text"]

    elif suffix == ".pdf":
        raise RuntimeError(
            "Use extract_pdf_text for PDF extraction."
        )

    else:
        text = source.read_text(
            encoding="utf-8",
            errors="replace",
        )

    destination.write_text(
        text,
        encoding="utf-8",
    )

    return {
        "success": True,
        "source": source_path,
        "output": str(destination),
        "format": "txt",
        "size": destination.stat().st_size,
    }


def _convert_to_docx(
    source: Path,
    destination: Path,
    source_path: str,
) -> dict[str, Any]:

    docx = require_dependency(
        "docx",
        "python-docx",
    )

    if source.suffix.lower() == ".docx":
        destination.write_bytes(
            source.read_bytes()
        )

    else:
        if source.suffix.lower() not in TEXT_EXTENSIONS:
            raise ValueError(
                "Only text-based files can currently "
                "be converted directly to DOCX."
            )

        text = source.read_text(
            encoding="utf-8",
            errors="replace",
        )

        document = docx.Document()

        for paragraph in text.splitlines():
            document.add_paragraph(
                paragraph
            )

        document.save(
            str(destination)
        )

    return {
        "success": True,
        "source": source_path,
        "output": str(destination),
        "format": "docx",
        "size": destination.stat().st_size,
    }

def _convert_to_pdf(
    source: Path,
    destination: Path,
    source_path: str,
) -> dict[str, Any]:

    if source.suffix.lower() in {
        ".html",
        ".htm",
    }:
        from .web import html_to_pdf

        return html_to_pdf(
            _WorkspaceProxy(source),
            source_path,
            str(destination),
        )

    raise ValueError(
        "PDF conversion currently supports HTML input. "
        "Use html_to_pdf for web content."
    )


class _WorkspaceProxy:
    """
    Internal adapter for conversion functions operating on
    an already-resolved file.
    """

    def __init__(
        self,
        path: Path,
    ) -> None:
        self._path = path

    def read_path(
        self,
        path: str,
    ) -> Path:
        return self._path

    def write_path(
        self,
        path: str,
    ) -> Path:
        return Path(path)