#!/usr/bin/env python3
"""
试卷文档读取模块：统一处理 docx / pdf / .doc，输出纯文本。

- docx：python-docx，保留段落+表格
- pdf：PyMuPDF (fitz)
- .doc（老格式）：mac textutil 转换；转不出（扫描件）则返回空并标记
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple


def read_paper(path: Path) -> Tuple[str, str]:
    """
    读取试卷，返回 (text, status)。
    status: ok / empty / unreadable / unsupported
    """
    path = Path(path)
    ext = path.suffix.lower()
    try:
        if ext == ".docx":
            return _read_docx(path)
        if ext == ".pdf":
            return _read_pdf(path)
        if ext == ".doc":
            return _read_doc(path)
        return "", "unsupported"
    except Exception as e:
        return f"[读取失败: {e}]", "unreadable"


def _read_docx(path: Path) -> Tuple[str, str]:
    import docx
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            cells = [c.text.strip() for c in row.cells]
            parts.append(" | ".join(cells))  # 表格行用 | 分隔，保留结构
    text = "\n".join(x for x in parts if x.strip())
    return text, ("ok" if len(text) > 200 else "empty")


def _read_pdf(path: Path) -> Tuple[str, str]:
    import fitz
    doc = fitz.open(str(path))
    text = "\n".join(page.get_text() for page in doc)
    return text, ("ok" if len(text) > 200 else "empty")


def _read_doc(path: Path) -> Tuple[str, str]:
    """老 .doc 用 mac textutil 转。扫描件转不出则 empty。"""
    try:
        out = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            capture_output=True, timeout=30,
        )
        text = out.stdout.decode("utf-8", errors="ignore")
        return text, ("ok" if len(text) > 200 else "empty")
    except FileNotFoundError:
        return "", "unsupported"  # 非 mac 环境
    except Exception as e:
        return f"[doc转换失败: {e}]", "unreadable"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text, status = read_paper(Path(sys.argv[1]))
        print(f"status={status}, 字数={len(text)}")
        print(text[:800])
