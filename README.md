# Medicord OCR Service

<div align="center">

**Production-grade OCR backend API for Cambodian medical documents**

Optimized for Khmer script (ខ្មែរ) + English medical terminology

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Tesseract](https://img.shields.io/badge/Tesseract-5.x_LSTM-4285F4?logo=google&logoColor=white)](https://github.com/tesseract-ocr/tesseract)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-Private-red)](LICENSE)

</div>

---

## Overview

Medicord OCR Service is a standalone, secure OCR microservice built for the [Medicord](https://github.com/jingf-dev) Flutter mobile app — a Cambodia-focused patient health records platform. It extracts and structures text from medical documents (lab reports, prescriptions, imaging results) with high accuracy for both Khmer and English content.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Engine OCR** | Tesseract 5 LSTM (primary) with automatic EasyOCR / PaddleOCR fallback when confidence < 75% |
| **Khmer + English** | Native support for mixed-script Cambodian hospital documents with Khmer numeral normalization (០-៩ ↔ 0-9) |
| **Image Preprocessing** | 5-stage OpenCV pipeline — grayscale → deskew → denoise → CLAHE → adaptive threshold |
| **PDF Support** | Multi-page PDF processing via poppler/pdf2image; large PDFs (>5 pages) auto-queued to background workers |
| **Structured Extraction** | Categorizes output into lab results, prescriptions, imaging reports with confidence scores |
| **Metadata Detection** | Regex-based extraction of report dates, facility names, and doctor names |
| **Async Processing** | Celery + Redis task queue for heavy workloads with progress tracking |
| **Security** | JWT bearer authentication, configurable rate limiting, auto-delete after processing |
| **Docker First** | Production-ready multi-service Docker Compose setup (API + Worker + Redis) |

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Framework | FastAPI + Uvicorn | 0.115.6 / 0.34.0 |
| Validation | Pydantic + pydantic-settings | 2.10.5 / 2.7.1 |
| Primary OCR | Tesseract 5 (LSTM engine) | 5.x |
| Fallback OCR | EasyOCR | 1.7.2 |
| Image Processing | OpenCV (headless) + NumPy + Pillow | 4.10.0 / 2.2.2 / 11.1.0 |
| PDF Conversion | pdf2image + poppler | 1.17.0 |
| Task Queue | Celery + Redis | 5.4.0 / 7-alpine |
| Authentication | python-jose (JWT) | 3.3.0 |
| Testing | pytest + pytest-asyncio | 8.3.4 / 0.25.3 |
| Runtime | Python 3.12-slim (Docker) | 3.12 |

---

## Project Structure

```
medicord-ocr-service/
├── app/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app factory with lifespan hooks
│   ├── config.py                      # Pydantic Settings (all env vars)
│   ├── api/
│   │   └── v1/
│   │       ├── router.py              # v1 route aggregator
│   │       └── endpoints/
│   │           ├── auth.py            # POST /auth/token
│   │           └── ocr.py             # POST /ocr/extract, GET /ocr/status
│   ├── auth/
│   │   └── jwt.py                     # JWT creation, validation, HTTPBearer
│   ├── middleware/
│   │   └── security.py                # Sliding-window rate limiter
│   ├── models/
│   │   └── schemas.py                 # Pydantic request/response schemas
│   ├── services/
│   │   ├── ocr_service.py             # Multi-engine OCR orchestrator
│   │   ├── preprocessing.py           # OpenCV 5-stage image pipeline
│   │   ├── pdf_service.py             # PDF → image conversion
│   │   └── postprocessing.py          # Regex categorization & structuring
│   ├── tasks/
│   │   ├── celery_app.py              # Celery broker/backend config
│   │   └── ocr_tasks.py              # Async OCR task definitions
│   └── utils/
│       └── khmer.py                   # Khmer digit/text normalization
├── tests/
│   └── test_ocr.py                    # Unit & integration tests (~20 cases)
├── Dockerfile                         # Python 3.12-slim, non-root, healthcheck
├── docker-compose.yml                 # API + Celery Worker + Redis
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose (recommended)
- _Or_ for local dev: Python 3.12+, Tesseract 5.x (`khm` + `eng` traineddata), Poppler, Redis

### Docker Setup (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/jingf-dev/medicord-ocr-service.git
cd medicord-ocr-service

# 2. Configure environment
cp .env.example .env
# Edit .env — set CLIENT_ID, CLIENT_SECRET, and JWT_SECRET_KEY

# 3. Build and start all services
docker compose up --build -d

# 4. Verify all containers are running
docker compose ps

# 5. Check the health endpoint
curl http://localhost:8080/health
```

> **Services started:** API (port 8080), Celery Worker, Redis (port 6379)

### Local Development

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
.\venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Start the server (with hot reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> **Note:** Ensure Tesseract, Poppler, and Redis are installed and available on your PATH for local development.

---

## API Reference

Base URL: `http://localhost:8080/api/v1`

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | No | Health check & version |
| `POST` | `/api/v1/auth/token` | No | Obtain JWT access token |
| `POST` | `/api/v1/ocr/extract` | Bearer | Extract text from document |
| `GET` | `/api/v1/ocr/status/{task_id}` | Bearer | Check async task progress |

Interactive API documentation is available at **http://localhost:8080/docs** (Swagger UI).

### 1. Obtain Access Token

```bash
curl -X POST http://localhost:8080/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "medicord-flutter-app",
    "client_secret": "your-secret-here"
  }'
```

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 2. Extract Text from Document

```bash
curl -X POST http://localhost:8080/api/v1/ocr/extract \
  -H "Authorization: Bearer <access_token>" \
  -F "file=@/path/to/medical-report.pdf" \
  -F "language_hint=khm+eng" \
  -F "patient_id=correlation-id-123"
```

**Successful Response (200):**

```json
{
  "raw_text": "...full OCR text...",
  "average_confidence": 0.93,
  "pages": 1,
  "file_type": "pdf",
  "extracted_at": "2026-02-19T14:30:26Z",
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
    "prescriptions": [
      {
        "medication": "Aspirin 81 mg",
        "dosage": "1 tablet",
        "frequency": "once daily",
        "duration": "30 days",
        "confidence": 0.0
      }
    ],
    "imaging": [],
    "other": []
  },
  "possible_report_date": "2026-01-20",
  "possible_facility": "Calmette Hospital",
  "possible_doctor": "Dr. Socheat"
}
```

**Large PDF Response (202 — queued for async processing):**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "message": "Large document queued for async processing."
}
```

### 3. Check Async Task Status

```bash
curl http://localhost:8080/api/v1/ocr/status/{task_id} \
  -H "Authorization: Bearer <access_token>"
```

### 4. Health Check

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

## Using Swagger UI

1. Open **http://localhost:8080/docs** in your browser
2. Call `POST /api/v1/auth/token` with your credentials to get a token
3. Click the **Authorize** 🔒 button (top-right)
4. Enter: `Bearer <your_access_token>` and click **Authorize**
5. Now all protected endpoints (like `/ocr/extract`) will include the token automatically

---

## Configuration

All settings are managed via environment variables. Copy `.env.example` to `.env` and customize:

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `medicord-ocr-service` | Application name |
| `APP_ENV` | `development` | Environment (`development` / `production`) |
| `DEBUG` | `true` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `CLIENT_ID` | `medicord-flutter-app` | OAuth2 client identifier |
| `CLIENT_SECRET` | — | **Required.** Client secret for token exchange |
| `JWT_SECRET_KEY` | — | **Required.** Secret key for JWT signing (min 32 chars) |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Token expiry in minutes |

### OCR Engine

| Variable | Default | Description |
|----------|---------|-------------|
| `TESSERACT_CMD` | `/usr/bin/tesseract` | Path to Tesseract binary |
| `TESSDATA_PREFIX` | `/usr/share/tesseract-ocr/5/tessdata` | Tesseract trained data directory |
| `DEFAULT_LANGUAGES` | `khm+eng` | Default OCR languages |
| `OCR_FALLBACK_ENGINE` | `easyocr` | Fallback engine (`easyocr` / `paddleocr`) |

### File Upload & Security

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FILE_SIZE_MB` | `10` | Maximum upload file size |
| `ALLOWED_EXTENSIONS` | `jpg,jpeg,png,pdf,tiff,bmp` | Accepted file types |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max requests per client per minute |
| `DELETE_AFTER_PROCESSING` | `true` | Auto-delete uploaded files after OCR |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `CELERY_BROKER_URL` | `redis://redis:6379/1` | Celery message broker |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/2` | Celery result backend |

### Optional: LLM Post-Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENABLED` | `false` | Enable LLM-based text correction |
| `LLM_PROVIDER` | `gemini` | LLM provider (`gemini` / `openai`) |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `OPENAI_API_KEY` | — | OpenAI API key |

---

## Architecture

```
┌──────────────┐      HTTPS / JSON       ┌─────────────────────────────────────┐
│  Flutter App  │ ──────────────────────► │          FastAPI (Uvicorn)          │
│   (Medicord)  │ ◄────────────────────── │         :8000 (container)           │
└──────────────┘    Structured JSON       │         :8080 (host)                │
                                          ├─────────────────────────────────────┤
                                          │  1. JWT Authentication              │
                                          │  2. File Validation (≤10 MB)        │
                                          │  3. Rate Limiting                   │
                                          ├─────────────────────────────────────┤
                                          │  4. PDF → Images (poppler)          │
                                          │  5. Preprocessing (OpenCV)          │
                                          │     ├─ Grayscale                    │
                                          │     ├─ Deskew (Hough lines)        │
                                          │     ├─ Denoise (fastNlMeans)       │
                                          │     ├─ CLAHE contrast enhance      │
                                          │     └─ Adaptive threshold           │
                                          ├─────────────────────────────────────┤
                                          │  6. OCR Engine                      │
                                          │     ├─ Tesseract 5 LSTM (primary)  │
                                          │     └─ EasyOCR (fallback < 75%)    │
                                          ├─────────────────────────────────────┤
                                          │  7. Post-Processing                 │
                                          │     ├─ Khmer normalization          │
                                          │     ├─ Regex categorization         │
                                          │     └─ Metadata extraction          │
                                          └────────────┬────────────────────────┘
                                                       │ Large PDFs (>5 pages)
                                                       ▼
                                          ┌─────────────────────────┐
                                          │   Celery Worker          │
                                          │   (async processing)     │
                                          └────────────┬────────────┘
                                                       │
                                                       ▼
                                          ┌─────────────────────────┐
                                          │   Redis 7                │
                                          │   (broker + results)     │
                                          └─────────────────────────┘
```

---

## Docker Services

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| `api` | `medicord-ocr-api` | 8080 → 8000 | FastAPI application server |
| `celery-worker` | `medicord-celery-worker` | — | Background OCR task processing |
| `redis` | `medicord-redis` | 6379 | Message broker & result backend |

### Useful Commands

```bash
# Start all services
docker compose up -d

# Rebuild after code changes
docker compose up --build -d

# View API logs
docker compose logs -f api

# View worker logs
docker compose logs -f celery-worker

# Stop all services
docker compose down

# Restart with fresh environment variables
docker compose down && docker compose up -d

# Shell into the API container
docker exec -it medicord-ocr-api bash
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage (if installed)
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Security Considerations

- **JWT Authentication** — All OCR endpoints require a valid Bearer token
- **File Cleanup** — Uploaded files are deleted immediately after processing (`DELETE_AFTER_PROCESSING=true`)
- **Size Limit** — Maximum upload size capped at 10 MB (configurable)
- **Rate Limiting** — Sliding-window rate limiter prevents abuse (30 req/min default)
- **Non-Root Container** — Docker runs as unprivileged `ocr` user
- **No Data Persistence** — OCR results are returned, not stored; no PII retention
- **CORS** — Configurable allowed origins (restrict in production)

---

## Khmer OCR Notes

- **Mixed Scripts** — Documents often contain Khmer (ខ្មែរ) and English on the same page; `khm+eng` language hint handles this
- **Khmer Numerals** — Automatic normalization between Khmer digits (០១២៣៤៥៦៧៨៩) and Arabic digits (0123456789)
- **Unicode Normalization** — NFC normalization + zero-width character removal for consistent text output
- **Complex Glyphs** — Khmer subscripts, vowels, and diacritics can be challenging; CLAHE + adaptive thresholding improves glyph separation
- **Font Sensitivity** — Best results with documents using standard Khmer fonts (Khmer OS, Noto Sans Khmer)

---

## Roadmap

- [ ] Fine-tune Tesseract LSTM with [KhmerSynthetic1M](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M) dataset
- [ ] Add TrOCR (microsoft/trocr-base) for handwritten document support
- [ ] LLM-powered post-processing for smarter text correction (Gemini Flash / Llama)
- [ ] Word-level confidence propagation to categorized results
- [ ] Medical document layout detection (table-aware OCR)
- [ ] GPU acceleration support for EasyOCR / PaddleOCR
- [ ] Prometheus metrics + structured logging

---

## License

Private — Medicord Project © 2026
