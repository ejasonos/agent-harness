from __future__ import annotations

from typing import Any

from workspace.manager import WorkspaceManager

from .engine import (
    resolve_input,
    resolve_output,
    require_dependency,
)


def extract_pdf_text(
    workspace: WorkspaceManager,
    path: str,
) -> dict[str, Any]:

    source = resolve_input(
        workspace,
        path,
    )

    fitz = require_dependency(
        "fitz",
        "PyMuPDF",
    )

    document = fitz.open(
        str(source)
    )

    pages: list[dict[str, Any]] = []
    combined: list[str] = []

    try:
        for index, page in enumerate(
            document
        ):
            text = page.get_text()

            pages.append(
                {
                    "page": index + 1,
                    "text": text,
                    "character_count": len(text),
                }
            )

            combined.append(text)

    finally:
        document.close()

    text = "\n".join(combined)

    return {
        "path": path,
        "format": "pdf",
        "page_count": len(pages),
        "text": text,
        "character_count": len(text),
        "pages": pages,
    }


def render_pdf_pages(
    workspace: WorkspaceManager,
    path: str,
    output_directory: str,
    start_page: int = 1,
    end_page: int | None = None,
    dpi: int = 150,
) -> dict[str, Any]:

    source = resolve_input(
        workspace,
        path,
    )

    output_dir = workspace.write_path(
        output_directory
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if start_page < 1:
        raise ValueError(
            "start_page must be >= 1."
        )

    if dpi < 50 or dpi > 600:
        raise ValueError(
            "dpi must be between 50 and 600."
        )

    fitz = require_dependency(
        "fitz",
        "PyMuPDF",
    )

    document = fitz.open(
        str(source)
    )

    try:
        total_pages = len(document)

        final_page = (
            total_pages
            if end_page is None
            else end_page
        )

        if final_page < start_page:
            raise ValueError(
                "end_page must be >= start_page."
            )

        if final_page > total_pages:
            raise ValueError(
                f"end_page cannot exceed {total_pages}."
            )

        zoom = dpi / 72

        matrix = fitz.Matrix(
            zoom,
            zoom,
        )

        rendered: list[str] = []

        for page_number in range(
            start_page,
            final_page + 1,
        ):

            page = document[
                page_number - 1
            ]

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            output_path = (
                output_dir
                / f"page-{page_number}.png"
            )

            pixmap.save(
                str(output_path)
            )

            rendered.append(
                str(output_path)
            )

    finally:
        document.close()

    return {
        "success": True,
        "source": path,
        "output_directory": str(output_dir),
        "pages": rendered,
        "page_count": len(rendered),
        "dpi": dpi,
    }


def create_pdf(
    workspace: WorkspaceManager,
    text: str,
    output_path: str,
) -> dict[str, Any]:

    destination = resolve_output(
        workspace,
        output_path,
    )

    reportlab = require_dependency(
        "reportlab",
    )

    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
    )
    from reportlab.lib.styles import (
        getSampleStyleSheet,
    )

    styles = getSampleStyleSheet()

    document = SimpleDocTemplate(
        str(destination),
        pagesize=letter,
    )

    story = []

    for line in text.splitlines():
        if line.strip():
            story.append(
                Paragraph(
                    line,
                    styles["BodyText"],
                )
            )

            story.append(
                Spacer(
                    1,
                    8,
                )
            )

    document.build(
        story
    )

    return {
        "success": True,
        "output": str(destination),
        "format": "pdf",
        "size": destination.stat().st_size,
    }