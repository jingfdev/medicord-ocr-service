"""Core OCR service – Tesseract primary, EasyOCR/PaddleOCR fallback.

Provides ensemble OCR with confidence-weighted merging for maximum
accuracy on Khmer + English medical documents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from app.config import get_settings
from app.services.preprocessing import ImagePreprocessor
from app.utils.khmer import normalize_khmer_text, khmer_digits_to_arabic

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Container for a single-page OCR result."""
    text: str
    confidence: float  # 0.0 – 1.0
    engine: str = "tesseract"
    word_confidences: List[float] = field(default_factory=list)


class OCRService:
    """Multi-engine OCR with preprocessing and confidence merging."""

    def __init__(self):
        self.settings = get_settings()
        self.preprocessor = ImagePreprocessor()
        self._easyocr_reader = None
        self._paddle_ocr = None

        # Configure Tesseract binary path
        pytesseract.pytesseract.tesseract_cmd = self.settings.TESSERACT_CMD

    # ──────────────── Public API ────────────────

    def extract_text(
        self,
        image: np.ndarray,
        languages: str = "",
        use_fallback: bool = True,
    ) -> OCRResult:
        """Extract text from a single image using ensemble OCR.

        Args:
            image: OpenCV BGR or grayscale image.
            languages: Tesseract language string (e.g. 'khm+eng').
            use_fallback: Whether to try fallback engine on low confidence.

        Returns:
            Best OCRResult from available engines.
        """
        languages = languages or self.settings.DEFAULT_LANGUAGES

        # Preprocess
        variants = self.preprocessor.process_for_ocr(image)

        # Primary: Tesseract
        tess_result = self._tesseract_ocr(variants[0], languages)
        logger.info(
            "Tesseract confidence=%.2f length=%d",
            tess_result.confidence,
            len(tess_result.text),
        )

        # If confidence is acceptable, return early
        if tess_result.confidence >= 0.75 or not use_fallback:
            return tess_result

        # Fallback engine
        logger.info("Low confidence (%.2f), trying fallback engine", tess_result.confidence)
        fallback_result = self._fallback_ocr(variants[1], languages)

        if fallback_result and fallback_result.confidence > tess_result.confidence:
            logger.info(
                "Fallback (%s) improved confidence: %.2f → %.2f",
                fallback_result.engine,
                tess_result.confidence,
                fallback_result.confidence,
            )
            return fallback_result

        return tess_result

    def extract_text_multi_page(
        self,
        images: List[np.ndarray],
        languages: str = "",
    ) -> Tuple[str, float, List[OCRResult]]:
        """Extract text from multiple page images.

        Returns:
            (combined_text, average_confidence, per_page_results)
        """
        results: List[OCRResult] = []
        for i, img in enumerate(images):
            logger.info("Processing page %d/%d", i + 1, len(images))
            result = self.extract_text(img, languages)
            results.append(result)

        combined_text = "\n\n--- Page Break ---\n\n".join(r.text for r in results)
        avg_confidence = (
            sum(r.confidence for r in results) / len(results) if results else 0.0
        )

        return combined_text, avg_confidence, results

    # ──────────────── Tesseract ────────────────

    def _tesseract_ocr(self, image: np.ndarray, languages: str) -> OCRResult:
        """Run Tesseract OCR with detailed confidence data."""
        try:
            # Get detailed data
            data = pytesseract.image_to_data(
                image,
                lang=languages,
                output_type=pytesseract.Output.DICT,
                config="--psm 6 --oem 1",  # PSM 6 = uniform block, OEM 1 = LSTM only
            )

            texts = []
            confidences = []
            for i, conf in enumerate(data["conf"]):
                conf_val = int(conf)
                word = data["text"][i].strip()
                if conf_val > 0 and word:
                    texts.append(word)
                    confidences.append(conf_val / 100.0)

            raw_text = " ".join(texts)
            raw_text = normalize_khmer_text(raw_text)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=raw_text,
                confidence=avg_conf,
                engine="tesseract",
                word_confidences=confidences,
            )
        except Exception as exc:
            logger.error("Tesseract failed: %s", exc)
            return OCRResult(text="", confidence=0.0, engine="tesseract")

    # ──────────────── Fallback: EasyOCR ────────────────

    def _get_easyocr_reader(self, languages: str):
        """Lazy-load EasyOCR reader."""
        if self._easyocr_reader is None:
            try:
                import easyocr

                lang_list = []
                if "khm" in languages or "km" in languages:
                    lang_list.append("km")  # EasyOCR uses ISO 639-1
                if "eng" in languages or "en" in languages:
                    lang_list.append("en")
                if not lang_list:
                    lang_list = ["en"]

                self._easyocr_reader = easyocr.Reader(
                    lang_list, gpu=False, verbose=False
                )
                logger.info("EasyOCR reader initialized: %s", lang_list)
            except ImportError:
                logger.warning("EasyOCR not installed – fallback unavailable")
                return None
        return self._easyocr_reader

    def _easyocr_extract(self, image: np.ndarray, languages: str) -> Optional[OCRResult]:
        """Run EasyOCR extraction."""
        reader = self._get_easyocr_reader(languages)
        if reader is None:
            return None

        try:
            results = reader.readtext(image, detail=1)
            texts = []
            confidences = []
            for _bbox, text, conf in results:
                texts.append(text)
                confidences.append(conf)

            raw_text = " ".join(texts)
            raw_text = normalize_khmer_text(raw_text)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=raw_text,
                confidence=avg_conf,
                engine="easyocr",
                word_confidences=confidences,
            )
        except Exception as exc:
            logger.error("EasyOCR failed: %s", exc)
            return None

    # ──────────────── Fallback: PaddleOCR ────────────────

    def _get_paddle_ocr(self, languages: str):
        """Lazy-load PaddleOCR."""
        if self._paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR

                lang = "en"
                if "khm" in languages or "km" in languages:
                    lang = "km"

                self._paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    show_log=False,
                )
                logger.info("PaddleOCR initialized: lang=%s", lang)
            except ImportError:
                logger.warning("PaddleOCR not installed – fallback unavailable")
                return None
        return self._paddle_ocr

    def _paddleocr_extract(self, image: np.ndarray, languages: str) -> Optional[OCRResult]:
        """Run PaddleOCR extraction."""
        ocr = self._get_paddle_ocr(languages)
        if ocr is None:
            return None

        try:
            result = ocr.ocr(image, cls=True)
            texts = []
            confidences = []
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    texts.append(text)
                    confidences.append(conf)

            raw_text = " ".join(texts)
            raw_text = normalize_khmer_text(raw_text)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=raw_text,
                confidence=avg_conf,
                engine="paddleocr",
                word_confidences=confidences,
            )
        except Exception as exc:
            logger.error("PaddleOCR failed: %s", exc)
            return None

    # ──────────────── Dispatch fallback ────────────────

    def _fallback_ocr(self, image: np.ndarray, languages: str) -> Optional[OCRResult]:
        """Try the configured fallback OCR engine."""
        engine = self.settings.OCR_FALLBACK_ENGINE.lower()

        if engine == "easyocr":
            return self._easyocr_extract(image, languages)
        elif engine == "paddleocr":
            return self._paddleocr_extract(image, languages)
        else:
            # Try EasyOCR first, then PaddleOCR
            result = self._easyocr_extract(image, languages)
            if result is None:
                result = self._paddleocr_extract(image, languages)
            return result
