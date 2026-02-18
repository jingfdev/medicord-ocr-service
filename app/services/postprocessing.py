"""Post-processing & text categorization for medical documents.

Extracts structured data from raw OCR text:
- Lab results (test name, value, unit, reference range)
- Prescriptions (medication, dosage, frequency, duration)
- Imaging reports (modality, findings, impression)
- Metadata: date, facility, doctor
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.models.schemas import (
    ExtractedCategories,
    ImagingResult,
    LabResult,
    OtherContent,
    Prescription,
)
from app.utils.khmer import (
    KHMER_MEDICAL_TERMS,
    khmer_digits_to_arabic,
    normalize_khmer_text,
    normalize_medical_value,
)

logger = logging.getLogger(__name__)


class PostProcessor:
    """Extract structured medical data from raw OCR text."""

    # ── Common lab test patterns ──
    LAB_PATTERNS = [
        # English pattern: Test Name ... Value Unit (Ref: range)
        re.compile(
            r"(?P<test>[\w\s\-/()]+?)\s*[:=]?\s*"
            r"(?P<value>\d+[.,]?\d*)\s*"
            r"(?P<unit>mg/dL|mmol/L|g/dL|%|U/L|mEq/L|µmol/L|ng/mL|pg/mL|IU/L|"
            r"cells/µL|×10[³⁹²]/[µu]L|mm/hr|sec|mL/min|fL|pg|g/L|mmHg)?\s*"
            r"(?:\(?\s*(?:ref|normal|range|N)[.:]?\s*(?P<ref>[\d.,–\-]+\s*[-–]\s*[\d.,–\-]+)\s*\)?)?",
            re.IGNORECASE,
        ),
        # Khmer pattern: test ... value
        re.compile(
            r"(?P<test>[\u1780-\u17FF\s]+)\s*[:=]?\s*"
            r"(?P<value>[០-៩\d]+[.,]?[០-៩\d]*)\s*"
            r"(?P<unit>\S+)?",
        ),
    ]

    # ── Prescription patterns ──
    RX_PATTERNS = [
        re.compile(
            r"(?P<med>[\w\s\-]+(?:\d+\s*(?:mg|mcg|g|ml|IU)))\s*"
            r"(?:[-–:]?\s*(?P<dosage>\d+\s*(?:tablet|cap|ml|dose|tab)s?))?[\s,]*"
            r"(?:(?P<freq>\d+\s*(?:times?|x)\s*(?:daily|per day|a day|/day)|"
            r"once daily|twice daily|BID|TID|QID|PRN|OD|BD|TDS|QDS))?\s*"
            r"(?:(?:for|x|×)\s*(?P<dur>\d+\s*(?:days?|weeks?|months?)))?",
            re.IGNORECASE,
        ),
    ]

    # ── Imaging modalities ──
    IMAGING_KEYWORDS = [
        "x-ray", "xray", "radiograph", "ultrasound", "echo", "echocardiogram",
        "ct scan", "ct", "mri", "mammogram", "fluoroscopy",
        "រូបភាពអេកូ", "អេកូ", "អ៊ិចស្រាយ",
    ]

    # ── Date patterns ──
    DATE_PATTERNS = [
        re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b"),
        re.compile(r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b"),
        re.compile(
            r"\b(\d{1,2})\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
            r"\s*(\d{4})\b",
            re.IGNORECASE,
        ),
    ]

    # ── Facility patterns ──
    FACILITY_PATTERNS = [
        re.compile(
            r"(?:hospital|clinic|centre|center|laboratory|lab|មន្ទីរពេទ្យ|គ្លីនិក)"
            r"[\s:]*(.+)",
            re.IGNORECASE,
        ),
        re.compile(r"([\w\s]+(?:Hospital|Clinic|Centre|Center|Laboratory))", re.IGNORECASE),
    ]

    # ── Doctor patterns ──
    DOCTOR_PATTERNS = [
        re.compile(r"(?:Dr\.?|Doctor|វេជ្ជបណ្ឌិត)[\s:]*([A-Za-z\u1780-\u17FF][\w\s\u1780-\u17FF]{2,30})", re.IGNORECASE),
    ]

    def process(self, raw_text: str, page_confidences: List[float] | None = None) -> dict:
        """Main entry: extract all structured data from OCR text.

        Returns dict compatible with OCRExtractionResponse fields.
        """
        text = normalize_khmer_text(raw_text)
        text_with_arabic = khmer_digits_to_arabic(text)

        categories = ExtractedCategories(
            lab_results=self._extract_lab_results(text_with_arabic),
            prescriptions=self._extract_prescriptions(text_with_arabic),
            imaging=self._extract_imaging(text),
            other=[],
        )

        return {
            "categories": categories,
            "possible_report_date": self._extract_date(text_with_arabic),
            "possible_facility": self._extract_facility(text),
            "possible_doctor": self._extract_doctor(text),
        }

    # ──────────────── Extractors ────────────────

    def _extract_lab_results(self, text: str) -> List[LabResult]:
        """Parse lab test results from text."""
        results: List[LabResult] = []
        seen_tests = set()

        for pattern in self.LAB_PATTERNS:
            for match in pattern.finditer(text):
                test_name = match.group("test").strip()
                if not test_name or test_name.lower() in seen_tests:
                    continue

                value_str = match.group("value")
                value = normalize_medical_value(value_str) if value_str else None

                unit = match.group("unit") if "unit" in match.groupdict() else None
                ref_range = match.group("ref") if "ref" in match.groupdict() else None

                # Determine if abnormal based on reference range
                abnormal = self._check_abnormal(value, ref_range)

                results.append(
                    LabResult(
                        test_name=test_name,
                        value=value,
                        unit=unit.strip() if unit else None,
                        reference_range=ref_range.strip() if ref_range else None,
                        abnormal=abnormal,
                        confidence=0.0,  # Will be filled by caller
                    )
                )
                seen_tests.add(test_name.lower())

        logger.info("Extracted %d lab results", len(results))
        return results

    def _extract_prescriptions(self, text: str) -> List[Prescription]:
        """Parse prescription / medication entries."""
        results: List[Prescription] = []

        for pattern in self.RX_PATTERNS:
            for match in pattern.finditer(text):
                med = match.group("med").strip()
                if len(med) < 3:
                    continue

                results.append(
                    Prescription(
                        medication=med,
                        dosage=self._clean(match, "dosage"),
                        frequency=self._clean(match, "freq"),
                        duration=self._clean(match, "dur"),
                        confidence=0.0,
                    )
                )

        logger.info("Extracted %d prescriptions", len(results))
        return results

    def _extract_imaging(self, text: str) -> List[ImagingResult]:
        """Detect imaging/radiology content."""
        results: List[ImagingResult] = []
        text_lower = text.lower()

        for keyword in self.IMAGING_KEYWORDS:
            if keyword.lower() in text_lower:
                # Find the section around the keyword
                idx = text_lower.index(keyword.lower())
                section = text[max(0, idx - 50): idx + 500]

                # Try to extract findings and impression
                findings = None
                impression = None

                findings_match = re.search(
                    r"(?:findings?|result|របាយការណ៍)[:\s]*(.+?)(?=impression|conclusion|$)",
                    section,
                    re.IGNORECASE | re.DOTALL,
                )
                if findings_match:
                    findings = findings_match.group(1).strip()[:500]

                impression_match = re.search(
                    r"(?:impression|conclusion|សន្និដ្ឋាន)[:\s]*(.+?)$",
                    section,
                    re.IGNORECASE | re.DOTALL,
                )
                if impression_match:
                    impression = impression_match.group(1).strip()[:500]

                results.append(
                    ImagingResult(
                        modality=keyword.title(),
                        body_part=None,
                        findings=findings,
                        impression=impression,
                        confidence=0.0,
                    )
                )
                break  # One imaging result per document usually

        return results

    def _extract_date(self, text: str) -> Optional[str]:
        """Try to find a report date in the text."""
        for pattern in self.DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                try:
                    # Try YYYY-MM-DD first
                    if len(groups[0]) == 4:
                        dt = datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                    else:
                        # DD/MM/YYYY
                        dt = datetime(int(groups[2]), int(groups[1]), int(groups[0]))
                    return dt.strftime("%Y-%m-%d")
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_facility(self, text: str) -> Optional[str]:
        """Try to find hospital / clinic name."""
        for pattern in self.FACILITY_PATTERNS:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if len(name) > 3:
                    return name[:100]
        return None

    def _extract_doctor(self, text: str) -> Optional[str]:
        """Try to find doctor name."""
        for pattern in self.DOCTOR_PATTERNS:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip()
                if len(name) > 2:
                    return f"Dr. {name}" if not name.startswith("Dr") else name
        return None

    # ──────────────── Helpers ────────────────

    @staticmethod
    def _check_abnormal(value: Optional[float], ref_range: Optional[str]) -> Optional[bool]:
        """Check if a value is outside the reference range."""
        if value is None or ref_range is None:
            return None

        # Parse range like "3.9-5.6" or "3.9–5.6"
        range_match = re.match(r"([\d.]+)\s*[-–]\s*([\d.]+)", ref_range)
        if not range_match:
            return None

        try:
            low = float(range_match.group(1))
            high = float(range_match.group(2))
            return value < low or value > high
        except ValueError:
            return None

    @staticmethod
    def _clean(match: re.Match, group: str) -> Optional[str]:
        """Safely extract and clean a named group from a regex match."""
        try:
            val = match.group(group)
            return val.strip() if val else None
        except (IndexError, AttributeError):
            return None
