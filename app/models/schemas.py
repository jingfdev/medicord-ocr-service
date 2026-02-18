"""Pydantic models / schemas for request & response payloads."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ────────────────────────── Enums ──────────────────────────

class FileType(str, Enum):
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    PDF = "pdf"
    TIFF = "tiff"
    BMP = "bmp"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ────────────────────────── Category items ──────────────────────────

class LabResult(BaseModel):
    """Single lab test result extracted from OCR."""
    test_name: str = Field(..., description="Test name (Khmer or English)")
    value: Optional[Union[float, str]] = Field(None, description="Numeric or text result")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    reference_range: Optional[str] = Field(None, description="Normal reference range")
    abnormal: Optional[bool] = Field(None, description="Whether the value is outside normal range")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Per-item OCR confidence")


class Prescription(BaseModel):
    """Single prescription item."""
    medication: str = Field(..., description="Drug name and strength")
    dosage: Optional[str] = Field(None, description="e.g. '1 tablet'")
    frequency: Optional[str] = Field(None, description="e.g. 'once daily'")
    duration: Optional[str] = Field(None, description="e.g. '30 days'")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ImagingResult(BaseModel):
    """Imaging / radiology report excerpt."""
    modality: Optional[str] = Field(None, description="e.g. X-ray, Ultrasound, CT")
    body_part: Optional[str] = Field(None, description="e.g. Chest, Abdomen")
    findings: Optional[str] = Field(None, description="Key findings text")
    impression: Optional[str] = Field(None, description="Radiologist impression")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class OtherContent(BaseModel):
    """Catch-all for content not fitting other categories."""
    section_title: Optional[str] = None
    content: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ExtractedCategories(BaseModel):
    """Structured categories extracted from the document."""
    lab_results: List[LabResult] = Field(default_factory=list)
    prescriptions: List[Prescription] = Field(default_factory=list)
    imaging: List[ImagingResult] = Field(default_factory=list)
    other: List[OtherContent] = Field(default_factory=list)


# ────────────────────────── Response models ──────────────────────────

class OCRExtractionResponse(BaseModel):
    """Main response from /ocr/extract endpoint."""
    raw_text: str = Field(..., description="Full OCR text output")
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean OCR confidence (0–1)")
    pages: int = Field(..., ge=1, description="Number of pages processed")
    file_type: str = Field(..., description="Detected file type")
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    categories: ExtractedCategories = Field(default_factory=ExtractedCategories)
    possible_report_date: Optional[str] = Field(None, description="Detected report date (ISO)")
    possible_facility: Optional[str] = Field(None, description="Detected hospital / clinic")
    possible_doctor: Optional[str] = Field(None, description="Detected physician name")


class AsyncTaskResponse(BaseModel):
    """Response when a task is queued for async processing."""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    message: str = "Task queued for processing"


class TaskStatusResponse(BaseModel):
    """Status check response for an async task."""
    task_id: str
    status: TaskStatus
    result: Optional[OCRExtractionResponse] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: Optional[str] = None


class TokenRequest(BaseModel):
    """Request body for token generation."""
    client_id: str
    client_secret: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
