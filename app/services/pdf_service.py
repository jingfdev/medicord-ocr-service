"""PDF → image conversion using pdf2image (poppler)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
from pdf2image import convert_from_path, convert_from_bytes

logger = logging.getLogger(__name__)

DEFAULT_DPI = 300


def pdf_to_images(
    pdf_path: str | Path,
    dpi: int = DEFAULT_DPI,
    fmt: str = "png",
) -> List[np.ndarray]:
    """Convert all pages of a PDF file to OpenCV-compatible numpy arrays.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering.
        fmt: Intermediate image format (png recommended for lossless).

    Returns:
        List of numpy arrays (BGR), one per page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Converting PDF to images: %s (dpi=%d)", pdf_path.name, dpi)

    pil_images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        fmt=fmt,
        thread_count=2,
    )

    images: List[np.ndarray] = []
    for i, pil_img in enumerate(pil_images):
        arr = np.array(pil_img)
        # PIL gives RGB, OpenCV expects BGR
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1].copy()
        images.append(arr)
        logger.debug("Page %d: %s", i + 1, arr.shape)

    logger.info("Converted %d pages from PDF", len(images))
    return images


def pdf_bytes_to_images(
    pdf_bytes: bytes,
    dpi: int = DEFAULT_DPI,
) -> List[np.ndarray]:
    """Convert PDF bytes (in-memory) to images."""
    pil_images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="png", thread_count=2)
    images = []
    for pil_img in pil_images:
        arr = np.array(pil_img)
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1].copy()
        images.append(arr)
    return images


def get_pdf_page_count(pdf_path: str | Path) -> int:
    """Get page count without fully converting."""
    from pdf2image.pdf2image import pdfinfo_from_path

    info = pdfinfo_from_path(str(pdf_path))
    return info.get("Pages", 0)
