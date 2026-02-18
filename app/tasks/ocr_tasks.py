"""Celery tasks for async OCR processing (multi-page PDFs)."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from celery import states

from app.tasks.celery_app import celery_app
from app.services.ocr_service import OCRService
from app.services.pdf_service import pdf_to_images
from app.services.postprocessing import PostProcessor
from app.config import get_settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="tasks.process_ocr")
def process_ocr_task(
    self,
    file_path: str,
    file_type: str,
    languages: str,
    patient_id: Optional[str] = None,
) -> dict:
    """Async OCR processing task for large documents.

    Args:
        file_path: Path to the uploaded file.
        file_type: Extension (pdf, jpg, png, etc.).
        languages: Tesseract language string.
        patient_id: Correlation ID for logging.

    Returns:
        Serialized OCRExtractionResponse dict.
    """
    settings = get_settings()
    logger.info(
        "Starting OCR task %s: file=%s type=%s patient=%s",
        self.request.id, file_path, file_type, patient_id,
    )

    self.update_state(state="PROCESSING", meta={"progress": 0})

    try:
        ocr_service = OCRService()
        post_processor = PostProcessor()

        # Load images
        if file_type == "pdf":
            images = pdf_to_images(file_path)
        else:
            import cv2
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Cannot read image: {file_path}")
            images = [img]

        total_pages = len(images)
        self.update_state(
            state="PROCESSING",
            meta={"progress": 10, "pages": total_pages},
        )

        # Run OCR on all pages
        combined_text, avg_confidence, page_results = ocr_service.extract_text_multi_page(
            images, languages
        )

        self.update_state(state="PROCESSING", meta={"progress": 70})

        # Post-process and categorize
        structured = post_processor.process(
            combined_text,
            [r.confidence for r in page_results],
        )

        self.update_state(state="PROCESSING", meta={"progress": 90})

        result = {
            "raw_text": combined_text,
            "average_confidence": round(avg_confidence, 4),
            "pages": total_pages,
            "file_type": file_type,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "categories": structured["categories"].model_dump(),
            "possible_report_date": structured["possible_report_date"],
            "possible_facility": structured["possible_facility"],
            "possible_doctor": structured["possible_doctor"],
        }

        logger.info(
            "Task %s completed: pages=%d confidence=%.2f",
            self.request.id, total_pages, avg_confidence,
        )

        return result

    except Exception as exc:
        logger.error("Task %s failed: %s", self.request.id, exc, exc_info=True)
        self.update_state(
            state=states.FAILURE,
            meta={"error": str(exc)},
        )
        raise

    finally:
        # Cleanup uploaded file
        if settings.DELETE_AFTER_PROCESSING:
            try:
                Path(file_path).unlink(missing_ok=True)
                logger.debug("Deleted file: %s", file_path)
            except Exception:
                pass
