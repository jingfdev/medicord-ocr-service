"""API v1 router – aggregates all v1 endpoint routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints.ocr import router as ocr_router
from app.api.v1.endpoints.auth import router as auth_router

api_v1_router = APIRouter()

api_v1_router.include_router(ocr_router)
api_v1_router.include_router(auth_router)
