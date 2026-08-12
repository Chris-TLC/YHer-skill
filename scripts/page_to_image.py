#!/usr/bin/env python3
"""
试卷 → 每页图片（视觉切题的前置）。

- PDF：PyMuPDF 直接渲染每页为 PNG（零额外依赖）。
- docx/.doc：先用 LibreOffice 转 PDF，再渲染。无 LibreOffice 则提示。

输出到 data/page_images/{卷名}/page_N.png
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

SKILL_DIR = Path(__file__).parent.parent
PAGE_IMG_DIR = SKILL_DIR / "data" / "page_images"

# LibreOffice 可执行路径（mac 常见位置）
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
    """docx/.doc → PDF（需 LibreOffice）。"""
    soffice = find_soffice()
    if not soffice:
        raise RuntimeError(
            "需要 LibreOffice 把 docx 转图。安装：brew install --cask libreoffice")
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(path)],
        capture_output=True, timeout=120, check=True,
    )
    pdf = out_dir / (path.stem + ".pdf")
    if not pdf.exists():
        raise RuntimeError(f"LibreOffice 转换失败: {path.name}")
    return pdf


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 150) -> List[Path]:
    """PDF 每页渲染为 PNG。"""
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
    """统一入口：试卷 → 页图片列表。"""
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
    raise ValueError(f"不支持的格式: {ext}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        imgs = paper_to_images(Path(sys.argv[1]))
        print(f"生成 {len(imgs)} 页图片:")
        for p in imgs:
            print(f"  {p}")
