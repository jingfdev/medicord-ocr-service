"""Security middleware – rate limiting, request validation."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import get_settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter.

    For production with multiple workers, use Redis-backed rate limiting
    (e.g. slowapi or a Redis Lua script). This in-memory version works
    well for single-instance / development.
    """

    def __init__(self, app):
        super().__init__(app)
        # {client_ip: [(timestamp, ...),]}
        self._requests: Dict[str, list] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        limit = settings.RATE_LIMIT_PER_MINUTE
        window = 60.0  # seconds

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Prune old entries
        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip] if now - ts < window
        ]

        if len(self._requests[client_ip]) >= limit:
            logger.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )

        self._requests[client_ip].append(now)
        response = await call_next(request)
        return response
