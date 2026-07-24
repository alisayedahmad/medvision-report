"""Per-key sliding window rate limiter.

In-memory, single-instance. Good enough until we need
Redis-backed limiting across replicas in Phase 7.
"""

import os
import time
from collections import defaultdict

from fastapi import HTTPException, Request


def _max_requests() -> int:
    return int(os.getenv("RATE_LIMIT_MAX", "60"))


def _window_seconds() -> float:
    return float(os.getenv("RATE_LIMIT_WINDOW", "60"))


# key -> list of request timestamps
_request_log: dict[str, list[float]] = defaultdict(list)


def _cleanup(key: str, now: float, window: float) -> None:
    """Drop timestamps outside the current window."""
    cutoff = now - window
    _request_log[key] = [t for t in _request_log[key] if t > cutoff]


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency — inject after auth so we have the key."""
    # Use the API key if present, fall back to client IP
    key = request.headers.get("X-API-Key") or request.client.host
    now = time.monotonic()
    window = _window_seconds()
    max_req = _max_requests()

    _cleanup(key, now, window)

    if len(_request_log[key]) >= max_req:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded, retry later",
        )

    _request_log[key].append(now)
