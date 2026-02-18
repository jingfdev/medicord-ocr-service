"""Khmer-specific text utilities.

Handles Khmer numeral conversion, script detection, common Unicode
normalization issues, and mixed Khmer/English text processing.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ── Khmer digit mapping ──
KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
ARABIC_DIGITS = "0123456789"
_KHMER_TO_ARABIC = str.maketrans(KHMER_DIGITS, ARABIC_DIGITS)
_ARABIC_TO_KHMER = str.maketrans(ARABIC_DIGITS, KHMER_DIGITS)

# ── Unicode ranges ──
KHMER_RANGE = (0x1780, 0x17FF)          # Main Khmer block
KHMER_SYMBOLS_RANGE = (0x19E0, 0x19FF)  # Khmer symbols


def khmer_digits_to_arabic(text: str) -> str:
    """Convert Khmer numerals ០-៩ to Arabic 0-9."""
    return text.translate(_KHMER_TO_ARABIC)


def arabic_digits_to_khmer(text: str) -> str:
    """Convert Arabic numerals 0-9 to Khmer ០-៩."""
    return text.translate(_ARABIC_TO_KHMER)


def normalize_khmer_text(text: str) -> str:
    """Normalize Khmer Unicode text.

    - NFC normalization
    - Fix common OCR substitutions
    - Normalize whitespace
    - Remove zero-width characters that break downstream parsing
    """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Remove zero-width chars (ZWSP, ZWJ, ZWNJ)  — OCR artefacts
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)

    # Normalize multiple spaces / tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize line endings
    text = re.sub(r"\r\n?", "\n", text)

    return text.strip()


def is_khmer_char(ch: str) -> bool:
    """Check if a single character is in the Khmer Unicode block."""
    cp = ord(ch)
    return (KHMER_RANGE[0] <= cp <= KHMER_RANGE[1]) or (
        KHMER_SYMBOLS_RANGE[0] <= cp <= KHMER_SYMBOLS_RANGE[1]
    )


def detect_script_ratio(text: str) -> dict:
    """Detect approximate ratio of Khmer vs Latin vs digits vs other."""
    stats = {"khmer": 0, "latin": 0, "digit": 0, "whitespace": 0, "other": 0}
    for ch in text:
        if ch.isspace():
            stats["whitespace"] += 1
        elif is_khmer_char(ch):
            stats["khmer"] += 1
        elif ch.isascii() and ch.isalpha():
            stats["latin"] += 1
        elif ch.isdigit() or ch in KHMER_DIGITS:
            stats["digit"] += 1
        else:
            stats["other"] += 1
    total = sum(stats.values()) or 1
    return {k: round(v / total, 3) for k, v in stats.items()}


def normalize_medical_value(text: str) -> Optional[float]:
    """Try to parse a numeric value from mixed Khmer/English OCR text.

    Handles:
    - Khmer digits
    - Common OCR errors: O→0, l→1, S→5
    - Decimal separators (. and ,)
    """
    # Convert Khmer digits first
    text = khmer_digits_to_arabic(text.strip())

    # Common OCR substitution fixes
    replacements = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"}
    cleaned = ""
    for ch in text:
        if ch in replacements and not ch.isalpha():
            cleaned += replacements[ch]
        elif ch.isdigit() or ch in ".,-":
            cleaned += ch

    # Normalize comma as decimal
    cleaned = cleaned.replace(",", ".")
    # Remove trailing dots/dashes
    cleaned = cleaned.strip(".-")

    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# ── Common Khmer medical terms mapping ──
KHMER_MEDICAL_TERMS = {
    "គ្លុយកូស": "Glucose",
    "ឈាម": "Blood",
    "ទឹកនោម": "Urine",
    "កម្រិត": "Level",
    "ធម្មតា": "Normal",
    "មិនធម្មតា": "Abnormal",
    "អ្នកជំងឺ": "Patient",
    "វេជ្ជបណ្ឌិត": "Doctor",
    "មន្ទីរពេទ្យ": "Hospital",
    "ថ្នាំ": "Medicine",
    "កម្រិតថ្នាំ": "Dosage",
    "រោគវិនិច្ឆ័យ": "Diagnosis",
    "លទ្ធផល": "Result",
    "ការពិនិត្យ": "Examination",
    "សម្ពាធឈាម": "Blood Pressure",
    "កម្ដៅខ្លួន": "Temperature",
    "ជំងឺទឹកនោមផ្អែម": "Diabetes",
    "សម្ពាធឈាមខ្ពស់": "Hypertension",
    "រូបភាពអេកូ": "Ultrasound",
    "កោសិកាឈាមក្រហម": "Red Blood Cells",
    "កោសិកាឈាមស": "White Blood Cells",
    "ប្លាកែត": "Platelets",
    "ហេម៉ូក្លូប៊ីន": "Hemoglobin",
}
