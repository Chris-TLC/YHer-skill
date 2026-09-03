#!/usr/bin/env python3
"""
Exam paper -> one image per page (prerequisite for visual item splitting).

- PDF: render each page to PNG directly with PyMuPDF (zero extra dependencies).
- docx/.doc: convert to PDF with LibreOffice first, then render. Warns if LibreOffice is missing.

Outputs to data/page_images/{paper_name}/page_N.png
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

SKILL_DIR = Path(__file__).parent.parent
PAGE_IMG_DIR = SKILL_DIR / "data" / "page_images"

# LibreOffice executable paths (common macOS locations)
SOFFICE_PATHS = [
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "soffice", "libreoffice",
]


def find_soffice():
    import shutil
    for p in SOFFICE_PATHS:
        if Path(p).exists() or shutil.which(p):
            return p
    return None


def docx_to_pdf(path: Path, out_dir: Path) -> Path:
    """docx/.doc -> PDF (requires LibreOffice)."""
    soffice = find_soffice()
    if not soffice:
        raise RuntimeError(
            "LibreOffice is required to convert docx files. Install with: brew install --cask libreoffice")
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(path)],
        capture_output=True, timeout=120, check=True,
    )
    pdf = out_dir / (path.stem + ".pdf")
    if not pdf.exists():
        raise RuntimeError(f"LibreOffice conversion failed: {path.name}")
    return pdf


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 150) -> List[Path]:
    """Render each PDF page as PNG."""
    import fitz
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        img_path = out_dir / f"page_{i+1:02d}.png"
        pix.save(str(img_path))
        images.append(img_path)
    return images


def paper_to_images(path: Path, dpi: int = 150) -> List[Path]:
    """Unified entry point: exam paper -> list of page images."""
    path = Path(path)
    out_dir = PAGE_IMG_DIR / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return pdf_to_images(path, out_dir, dpi)
    if ext in (".docx", ".doc"):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = docx_to_pdf(path, Path(tmp))
            return pdf_to_images(pdf, out_dir, dpi)
    raise ValueError(f"Unsupported format: {ext}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        imgs = paper_to_images(Path(sys.argv[1]))
        print(f"Generated {len(imgs)} page images:")
        for p in imgs:
            print(f"  {p}")
