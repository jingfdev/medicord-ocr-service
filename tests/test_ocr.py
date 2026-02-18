"""Tests for the OCR extraction API."""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.auth.jwt import create_access_token
from app.utils.khmer import (
    khmer_digits_to_arabic,
    normalize_khmer_text,
    detect_script_ratio,
    normalize_medical_value,
)
from app.services.preprocessing import ImagePreprocessor
from app.services.postprocessing import PostProcessor


# ── Helpers ──

def _get_auth_header() -> dict:
    """Create a valid JWT auth header for tests."""
    token = create_access_token(data={"sub": "test-client", "type": "client"})
    return {"Authorization": f"Bearer {token}"}


def _create_test_image(width: int = 800, height: int = 600) -> bytes:
    """Create a simple test image with text-like patterns."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # white
    # Add some black rectangles to simulate text areas
    for y in range(100, 500, 40):
        img[y : y + 15, 50:750] = 0
    # Encode as PNG
    import cv2
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ── Khmer utility tests ──

class TestKhmerUtils:
    def test_khmer_to_arabic_digits(self):
        assert khmer_digits_to_arabic("០១២៣៤៥៦៧៨៩") == "0123456789"

    def test_mixed_text_digits(self):
        result = khmer_digits_to_arabic("គ្លុយកូស: ៥.៨ mmol/L")
        assert "5.8" in result

    def test_normalize_text(self):
        text = "  hello\u200B  world\u200C  "
        result = normalize_khmer_text(text)
        assert "\u200B" not in result
        assert "\u200C" not in result
        assert result == "hello world"

    def test_detect_script_ratio(self):
        result = detect_script_ratio("Hello World 123")
        assert result["latin"] > 0
        assert result["digit"] > 0

    def test_detect_khmer_script(self):
        result = detect_script_ratio("គ្លុយកូស ១២៣")
        assert result["khmer"] > 0

    def test_normalize_medical_value(self):
        assert normalize_medical_value("5.8") == 5.8
        assert normalize_medical_value("៥.៨") == 5.8
        assert normalize_medical_value("12,5") == 12.5
        assert normalize_medical_value("abc") is None


# ── Preprocessing tests ──

class TestPreprocessor:
    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_grayscale_conversion(self):
        # BGR image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = self.preprocessor._to_grayscale(img)
        assert len(gray.shape) == 2

    def test_already_grayscale(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        gray = self.preprocessor._to_grayscale(img)
        assert len(gray.shape) == 2

    def test_ensure_min_size(self):
        small = np.ones((100, 100, 3), dtype=np.uint8)
        result = self.preprocessor._ensure_min_size(small, min_height=500)
        assert result.shape[0] >= 500

    def test_adaptive_threshold(self):
        gray = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = self.preprocessor._adaptive_threshold(gray)
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_full_process(self):
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        result = self.preprocessor.process(img)
        assert result.shape[0] > 0

    def test_process_for_ocr_returns_variants(self):
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        variants = self.preprocessor.process_for_ocr(img)
        assert len(variants) == 3


# ── Post-processor tests ──

class TestPostProcessor:
    def setup_method(self):
        self.processor = PostProcessor()

    def test_extract_date_ddmmyyyy(self):
        text = "Report Date: 20/01/2026"
        result = self.processor._extract_date(text)
        assert result == "2026-01-20"

    def test_extract_date_yyyymmdd(self):
        text = "Date: 2026-01-20"
        result = self.processor._extract_date(text)
        assert result == "2026-01-20"

    def test_extract_facility(self):
        text = "Calmette Hospital\nPhnom Penh, Cambodia"
        result = self.processor._extract_facility(text)
        assert result is not None
        assert "Calmette" in result

    def test_extract_doctor(self):
        text = "Dr. Socheat Nhem\nCardiologist"
        result = self.processor._extract_doctor(text)
        assert result is not None
        assert "Socheat" in result

    def test_check_abnormal_high(self):
        assert self.processor._check_abnormal(6.0, "3.9-5.6") is True

    def test_check_abnormal_normal(self):
        assert self.processor._check_abnormal(4.5, "3.9-5.6") is False

    def test_check_abnormal_low(self):
        assert self.processor._check_abnormal(3.0, "3.9-5.6") is True

    def test_check_abnormal_no_range(self):
        assert self.processor._check_abnormal(5.0, None) is None

    def test_full_process(self):
        text = (
            "Calmette Hospital\n"
            "Date: 2026-01-20\n"
            "Dr. Socheat\n"
            "Fasting Glucose: 5.8 mmol/L (ref: 3.9-5.6)\n"
            "Aspirin 81 mg 1 tablet once daily for 30 days\n"
        )
        result = self.processor.process(text)
        assert result["possible_facility"] is not None
        assert result["possible_doctor"] is not None
        assert result["categories"] is not None


# ── API endpoint tests ──

class TestAPI:
    def setup_method(self):
        self.client = TestClient(app)
        self.auth = _get_auth_header()

    def test_health_check(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_extract_no_auth(self):
        response = self.client.post("/api/v1/ocr/extract")
        assert response.status_code in (401, 403)

    def test_extract_no_file(self):
        response = self.client.post(
            "/api/v1/ocr/extract",
            headers=self.auth,
        )
        assert response.status_code == 422  # missing required file

    def test_extract_invalid_extension(self):
        response = self.client.post(
            "/api/v1/ocr/extract",
            headers=self.auth,
            files={"file": ("test.exe", b"data", "application/octet-stream")},
        )
        assert response.status_code == 400

    @patch("app.api.v1.endpoints.ocr.OCRService")
    def test_extract_valid_image(self, mock_ocr_cls):
        """Test with a valid PNG image (mocked OCR)."""
        from app.services.ocr_service import OCRResult

        mock_service = MagicMock()
        mock_service.extract_text_multi_page.return_value = (
            "Test text",
            0.85,
            [OCRResult(text="Test text", confidence=0.85, engine="tesseract")],
        )
        mock_ocr_cls.return_value = mock_service

        img_bytes = _create_test_image()
        response = self.client.post(
            "/api/v1/ocr/extract",
            headers=self.auth,
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"language_hint": "khm+eng"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "raw_text" in data
        assert "average_confidence" in data
        assert "categories" in data

    def test_auth_token_endpoint(self):
        response = self.client.post(
            "/api/v1/auth/token",
            json={
                "client_id": "medicord-flutter-app",
                "client_secret": "change-me-to-a-secure-secret",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_auth_token_invalid(self):
        response = self.client.post(
            "/api/v1/auth/token",
            json={"client_id": "invalid", "client_secret": "wrong"},
        )
        assert response.status_code == 401
