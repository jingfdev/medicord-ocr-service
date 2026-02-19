"""Authentication endpoint for generating JWT tokens."""

from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, HTTPException, status

from app.auth.jwt import create_access_token
from app.config import get_settings
from app.models.schemas import TokenRequest, TokenResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authentication"])


@router.post(
    "/auth/token",
    response_model=TokenResponse,
    summary="Generate access token",
    description="Authenticate with client credentials to receive a JWT bearer token.",
)
async def create_token(body: TokenRequest):
    """Issue a JWT token for a valid client."""
    settings = get_settings()

    if body.client_id != settings.CLIENT_ID or body.client_secret != settings.CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
        )

    expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    token = create_access_token(
        data={"sub": body.client_id, "type": "client"},
        expires_delta=expires,
    )

    logger.info("Token issued for client: %s", body.client_id)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=int(expires.total_seconds()),
    )
