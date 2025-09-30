#!/usr/bin/env python3
"""
AIVA Middleware
Request logging and rate limiting middleware
"""

import time
from typing import Dict, List
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from .config import LOG_REQUESTS, LOG_MAX_BODY, RATE_LIMIT_PER_MIN

# Rate limiting storage
_rl_counters: Dict[str, List[float]] = {}

def redact_body(body: bytes) -> str:
    """Redact sensitive information from request body for logging"""
    try:
        s = body.decode("utf-8", errors="replace")
    except Exception:
        return "<non-text>"

    if len(s) > LOG_MAX_BODY:
        s = s[:LOG_MAX_BODY] + "…<truncated>"

    return s

async def logging_and_rate_limit_middleware(request: Request, call_next) -> Response:
    """
    Middleware for request logging and rate limiting

    Args:
        request: FastAPI request object
        call_next: Next middleware/route handler

    Returns:
        Response object
    """
    # Rate limit per client IP
    ip = (request.client.host if request.client else "?")
    now = time.time()
    window_start = now - 60.0
    buf = _rl_counters.get(ip, [])
    buf = [t for t in buf if t >= window_start]

    if len(buf) >= RATE_LIMIT_PER_MIN:
        return StreamingResponse(
            iter([b'{"error":"rate_limited"}']),
            status_code=429,
            media_type="application/json"
        )

    buf.append(now)
    _rl_counters[ip] = buf

    # Logging
    if LOG_REQUESTS:
        try:
            body = await request.body()
        except Exception:
            body = b""

        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"REQ {ip} {request.method} {request.url.path} "
            f"UA={request.headers.get('user-agent','-')} "
            f"BODY={redact_body(body)}"
        )

    response = await call_next(request)
    return response