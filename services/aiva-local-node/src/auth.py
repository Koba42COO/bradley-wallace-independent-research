#!/usr/bin/env python3
"""
AIVA Authentication
Authentication and authorization utilities
"""

from fastapi import HTTPException, Request
from .config import API_KEY

def verify_auth(req: Request) -> None:
    """
    Verify API authentication

    Args:
        req: FastAPI request object

    Raises:
        HTTPException: If authentication fails
    """
    if not API_KEY:
        return  # auth disabled

    # Check Authorization header
    hdr = req.headers.get("authorization") or req.headers.get("Authorization")
    xkey = req.headers.get("x-api-key") or req.headers.get("X-API-Key")

    token = None
    if hdr and hdr.lower().startswith("bearer "):
        token = hdr.split(" ", 1)[1].strip()
    elif xkey:
        token = xkey.strip()

    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")