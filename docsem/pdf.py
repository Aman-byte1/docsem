"""PDF utilities: page rendering with PyMuPDF.

The DocSem PDFs are full-page scans (no text layer), so we render each page to a
PNG and OCR it (VLM on the A100, or rapidocr locally).
"""
from __future__ import annotations

from pathlib import Path

import pymupdf


def page_count(pdf_path: str | Path) -> int:
    with pymupdf.open(str(pdf_path)) as doc:
        return len(doc)


def render_pdf_pages(pdf_path: str | Path, out_dir: str | Path, dpi: int = 160) -> list[Path]:
    """Render every page of a PDF to `out_dir/p01.png, p02.png, ...` and return the paths.

    Returns already-rendered pages if they exist (resume-friendly).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i in range(len(doc)):
            png = out_dir / f"p{i + 1:02d}.png"
            if not png.exists():
                pix = doc[i].get_pixmap(dpi=dpi)
                pix.save(str(png))
            paths.append(png)
    return paths


def render_page_image(pdf_path: str | Path, page_index: int, dpi: int = 200) -> Path:
    """Render a single page to a temp PNG (used by the rapidocr fallback)."""
    import tempfile

    with pymupdf.open(str(pdf_path)) as doc:
        pix = doc[page_index].get_pixmap(dpi=dpi)
        tmp = Path(tempfile.gettempdir()) / f"docsem_page_{abs(hash(str(pdf_path)))}_{page_index}.png"
        pix.save(str(tmp))
        return tmp