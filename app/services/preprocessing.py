"""Image preprocessing pipeline using OpenCV.

Optimized for Cambodian medical documents: lab reports, prescriptions,
ultrasound/imaging reports. Handles deskew, denoising, contrast
enhancement, and adaptive thresholding.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Full preprocessing pipeline for OCR-ready images."""

    def __init__(
        self,
        target_dpi: int = 300,
        denoise_strength: int = 10,
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (8, 8),
    ):
        self.target_dpi = target_dpi
        self.denoise_strength = denoise_strength
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run full preprocessing pipeline on a single image.

        Steps:
            1. Resize to target DPI (if too small)
            2. Convert to grayscale
            3. Deskew
            4. Denoise
            5. CLAHE contrast enhancement
            6. Adaptive thresholding
            7. Remove borders / noise artefacts
        """
        logger.debug("Preprocessing image: shape=%s dtype=%s", image.shape, image.dtype)

        img = self._ensure_min_size(image)
        gray = self._to_grayscale(img)
        deskewed = self._deskew(gray)
        denoised = self._denoise(deskewed)
        enhanced = self._clahe_enhance(denoised)
        binarized = self._adaptive_threshold(enhanced)
        cleaned = self._remove_border_noise(binarized)

        logger.debug("Preprocessing complete: shape=%s", cleaned.shape)
        return cleaned

    def process_for_ocr(self, image: np.ndarray) -> List[np.ndarray]:
        """Return multiple variants for ensemble OCR.

        Returns:
            [binary, enhanced_gray, original_gray] — allows different
            OCR engines to pick the best input.
        """
        gray = self._to_grayscale(image)
        deskewed = self._deskew(gray)
        denoised = self._denoise(deskewed)
        enhanced = self._clahe_enhance(denoised)
        binary = self._adaptive_threshold(enhanced)
        return [binary, enhanced, deskewed]

    # ──────────────── Individual steps ────────────────

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image

    def _ensure_min_size(self, image: np.ndarray, min_height: int = 1000) -> np.ndarray:
        """Up-scale small images so that glyphs are large enough for OCR."""
        h, w = image.shape[:2]
        if h < min_height:
            scale = min_height / h
            image = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            logger.debug("Upscaled image %.1fx → %s", scale, image.shape[:2])
        return image

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Correct rotation using Hough line detection."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )
        if lines is None:
            return gray

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines
            if abs(angle) < 15:
                angles.append(angle)

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.3:
            return gray  # already straight

        logger.debug("Deskewing by %.2f°", median_angle)
        h, w = gray.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            gray, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        """Non-local means denoising — good for preserving Khmer diacritics."""
        return cv2.fastNlMeansDenoising(
            gray, None, h=self.denoise_strength, templateWindowSize=7, searchWindowSize=21
        )

    def _clahe_enhance(self, gray: np.ndarray) -> np.ndarray:
        """Contrast-Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        return clahe.apply(gray)

    @staticmethod
    def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
        """Binarize with adaptive Gaussian thresholding."""
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=11
        )

    @staticmethod
    def _remove_border_noise(binary: np.ndarray, margin: int = 5) -> np.ndarray:
        """Flood-fill corners to remove scanner borders / shadows."""
        h, w = binary.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        for x, y in corners:
            if binary[y, x] == 0:
                cv2.floodFill(binary, mask, (x, y), 255)
        return binary

    @staticmethod
    def sharpen(gray: np.ndarray) -> np.ndarray:
        """Optional sharpening kernel."""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(gray, -1, kernel)

    @staticmethod
    def morphological_clean(binary: np.ndarray) -> np.ndarray:
        """Remove small noise blobs via morphological opening."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
