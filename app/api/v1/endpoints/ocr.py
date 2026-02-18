"""OCR extraction endpoint."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.auth.jwt import get_current_user
from app.config import get_settings
from app.models.schemas import (
    AsyncTaskResponse,
    ErrorResponse,
    OCRExtractionResponse,
    TaskStatus,
    TaskStatusResponse,
)
from app.services.ocr_service import OCRService
from app.services.pdf_service import pdf_to_images, get_pdf_page_count
from app.services.postprocessing import PostProcessor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OCR"])


def _validate_file(file: UploadFile) -> str:
    """Validate file extension and size. Returns normalized extension."""
    settings = get_settings()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )
    return ext


async def _save_upload(file: UploadFile, ext: str) -> tuple[Path, bytes]:
    """Save uploaded file to temp directory & return path + raw bytes."""
    settings = get_settings()
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.{ext}"
    file_path = upload_dir / filename

    content = await file.read()

    # Check file size
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB} MB",
        )

    file_path.write_bytes(content)
    return file_path, content


@router.post(
    "/ocr/extract",
    response_model=OCRExtractionResponse,
    responses={
        202: {"model": AsyncTaskResponse, "description": "Task queued (large PDF)"},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    summary="Extract structured data from medical document",
    description=(
        "Upload a medical document image (JPG/PNG) or multi-page PDF. "
        "Returns structured OCR results categorized into lab results, "
        "prescriptions, imaging reports, with confidence scores and metadata."
    ),
)
async def extract_ocr(
    file: UploadFile = File(..., description="Medical document image or PDF"),
    language_hint: str = Form(default="khm+eng", description="OCR language hint"),
    patient_id: Optional[str] = Form(default=None, description="Correlation ID (not stored)"),
    _user: dict = Depends(get_current_user),
):
    """Main OCR extraction endpoint."""
    settings = get_settings()

    # Validate
    ext = _validate_file(file)
    file_path, content = await _save_upload(file, ext)

    logger.info(
        "OCR request: file=%s size=%d lang=%s patient=%s",
        file.filename, len(content), language_hint, patient_id,
    )

    try:
        # Load images
        if ext == "pdf":
            page_count = get_pdf_page_count(file_path)

            # For large PDFs, queue async task
            if page_count > 5:
                return await _queue_async_task(str(file_path), ext, language_hint, patient_id)

            images = pdf_to_images(file_path)
        else:
            # Decode image from bytes
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Cannot decode image file")
            images = [img]

        # Run OCR
        ocr_service = OCRService()
        combined_text, avg_confidence, page_results = ocr_service.extract_text_multi_page(
            images, language_hint
        )

        # Post-process
        post_processor = PostProcessor()
        structured = post_processor.process(
            combined_text,
            [r.confidence for r in page_results],
        )

        # Build response
        response = OCRExtractionResponse(
            raw_text=combined_text,
            average_confidence=round(avg_confidence, 4),
            pages=len(images),
            file_type=ext,
            extracted_at=datetime.now(timezone.utc),
            categories=structured["categories"],
            possible_report_date=structured["possible_report_date"],
            possible_facility=structured["possible_facility"],
            possible_doctor=structured["possible_doctor"],
        )

        logger.info(
            "OCR complete: pages=%d confidence=%.2f labs=%d rx=%d",
            len(images),
            avg_confidence,
            len(response.categories.lab_results),
            len(response.categories.prescriptions),
        )

        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("OCR processing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR processing error: {exc}")

    finally:
        # Always clean up uploaded file
        if settings.DELETE_AFTER_PROCESSING and file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass


async def _queue_async_task(
    file_path: str, file_type: str, languages: str, patient_id: str | None
) -> AsyncTaskResponse:
    """Queue a Celery task for large documents."""
    try:
        from app.tasks.ocr_tasks import process_ocr_task

        task = process_ocr_task.delay(
            file_path=file_path,
            file_type=file_type,
            languages=languages,
            patient_id=patient_id,
        )

        logger.info("Queued async task: %s", task.id)

        return AsyncTaskResponse(
            task_id=task.id,
            status=TaskStatus.PENDING,
            message="Large document queued for async processing. Check status at /api/v1/ocr/status/{task_id}",
        )
    except Exception as exc:
        logger.error("Failed to queue task: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Async processing unavailable. Ensure Redis/Celery are running.",
        )


@router.get(
    "/ocr/status/{task_id}",
    response_model=TaskStatusResponse,
    summary="Check async OCR task status",
)
async def get_task_status(
    task_id: str,
    _user: dict = Depends(get_current_user),
):
    """Check the status of an async OCR processing task."""
    try:
        from app.tasks.celery_app import celery_app

        result = celery_app.AsyncResult(task_id)

        if result.state == "PENDING":
            return TaskStatusResponse(task_id=task_id, status=TaskStatus.PENDING)
        elif result.state == "PROCESSING":
            return TaskStatusResponse(task_id=task_id, status=TaskStatus.PROCESSING)
        elif result.state == "SUCCESS":
            return TaskStatusResponse(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=OCRExtractionResponse(**result.result),
            )
        else:
            error_msg = str(result.info) if result.info else "Unknown error"
            return TaskStatusResponse(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error checking task: {exc}")
