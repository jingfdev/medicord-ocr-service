# Medicord OCR Service

> High-accuracy OCR backend API for Cambodian medical documents — optimized for Khmer script + English medical terminology.

Built with **FastAPI**, **Tesseract 5 (LSTM)**, **EasyOCR** fallback, **OpenCV** preprocessing, and **Celery + Redis** for async processing.

---

## Features

- **Multi-engine OCR**: Tesseract 5 (primary) with EasyOCR/PaddleOCR fallback for maximum accuracy
- **Khmer + English**: Optimized for mixed-script Cambodian medical documents
- **Image preprocessing**: Deskew, denoise, CLAHE contrast enhancement, adaptive thresholding
- **PDF support**: Multi-page PDF conversion via poppler/pdf2image
- **Structured extraction**: Lab results, prescriptions, imaging reports with confidence scores
- **Metadata detection**: Automatic date, facility, and doctor name extraction
- **Async processing**: Celery + Redis for large PDFs (>5 pages)
- **JWT authentication**: Bearer token security
- **Rate limiting**: Configurable per-minute request limit
- **Docker ready**: Production Dockerfile + docker-compose with Redis

---

## Project Structure

```
medicord-ocr-service/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application factory
│   ├── config.py                  # Settings (env vars)
│   ├── api/
│   │   └── v1/
│   │       ├── router.py          # v1 router aggregator
│   │       └── endpoints/
│   │           ├── ocr.py         # POST /ocr/extract & GET /ocr/status
│   │           └── auth.py        # POST /auth/token
│   ├── auth/
│   │   └── jwt.py                 # JWT token creation & validation
│   ├── middleware/
│   │   └── security.py            # Rate limiting middleware
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   ├── services/
│   │   ├── ocr_service.py         # Multi-engine OCR orchestration
│   │   ├── preprocessing.py       # OpenCV image preprocessing pipeline
│   │   ├── pdf_service.py         # PDF → image conversion
│   │   └── postprocessing.py      # Text categorization & structuring
│   ├── tasks/
│   │   ├── celery_app.py          # Celery configuration
│   │   └── ocr_tasks.py           # Async OCR tasks
│   └── utils/
│       └── khmer.py               # Khmer text utilities & normalization
├── tests/
│   └── test_ocr.py                # Unit & integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## Quick Start

### 1. Prerequisites

- Python 3.12+
- Tesseract 5.x with `khm` and `eng` trained data
- Poppler (for PDF support)
- Redis (for async tasks)

### 2. Local Development

```bash
# Clone and setup
git clone <repo-url>
cd medicord-ocr-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Docker (Recommended)

```bash
# Copy and configure environment
cp .env.example .env

# Build and start all services
docker compose up --build -d

# Check logs
docker compose logs -f api
```

The API will be available at `http://localhost:8000`.

---

## API Endpoints

### Authentication

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "client_id": "medicord-flutter-app",
  "client_secret": "your-secret"
}
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### OCR Extraction

```http
POST /api/v1/ocr/extract
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <medical document image or PDF>
language_hint: khm+eng
patient_id: correlation-id-123
```

**Response (200):**
```json
{
  "raw_text": "...full OCR string...",
  "average_confidence": 0.84,
  "pages": 1,
  "file_type": "png",
  "extracted_at": "2026-02-18T15:45:22Z",
  "categories": {
    "lab_results": [
      {
        "test_name": "Fasting Glucose",
        "value": 5.8,
        "unit": "mmol/L",
        "reference_range": "3.9–5.6",
        "abnormal": true,
        "confidence": 0.91
      }
    ],
    "prescriptions": [...],
    "imaging": [...],
    "other": [...]
  },
  "possible_report_date": "2026-01-20",
  "possible_facility": "Calmette Hospital",
  "possible_doctor": "Dr. Socheat"
}
```

**Response (202 — large PDF queued):**
```json
{
  "task_id": "abc-123",
  "status": "pending",
  "message": "Large document queued for async processing."
}
```

### Check Async Task Status

```http
GET /api/v1/ocr/status/{task_id}
Authorization: Bearer <token>
```

### Health Check

```http
GET /health
→ {"status": "healthy", "version": "1.0.0"}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

All settings are configured via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET_KEY` | (change me) | Secret key for JWT signing |
| `DEFAULT_LANGUAGES` | `khm+eng` | Tesseract language string |
| `OCR_FALLBACK_ENGINE` | `easyocr` | Fallback OCR engine |
| `MAX_FILE_SIZE_MB` | `10` | Maximum upload size |
| `RATE_LIMIT_PER_MINUTE` | `30` | API rate limit per client |
| `DELETE_AFTER_PROCESSING` | `true` | Delete uploads after OCR |
| `CELERY_BROKER_URL` | `redis://...` | Celery broker |
| `LLM_ENABLED` | `false` | Enable LLM post-processing |

---

## Architecture

```
Flutter App → POST /ocr/extract → FastAPI
                                    ├── Validate & save file
                                    ├── PDF? → pdf2image (poppler)
                                    ├── Preprocess (OpenCV pipeline)
                                    │   ├── Grayscale → Deskew → Denoise
                                    │   └── CLAHE → Adaptive threshold
                                    ├── OCR Engine
                                    │   ├── Tesseract 5 LSTM (primary)
                                    │   └── EasyOCR / PaddleOCR (fallback)
                                    ├── Post-process
                                    │   ├── Khmer normalization
                                    │   ├── Regex extraction
                                    │   └── Categorization
                                    └── Return structured JSON
```

---

## License

Private — Medicord Project
