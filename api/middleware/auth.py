"""API key authentication dependency.

Checks X-API-Key header against keys from environment
Uses constant-time comparison to prevent timing attacks

"""

import os
import secrets

from fastapi import Depends, HTTPException, Security

from fastapi.security import APIKeyHeader

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _auth_enabled() -> bool:
    return os.getenv("AUTH_ENABLED", "true").lower() != "false"


def _valid_keys() -> set[str]:
    raw = os.getenv("MEDVISION_API_KEY", "")
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def _is_valid_key(provided: str, valid: set[str]) -> bool:

    # Compare against every key so timing does not reveal which one matched
    return any(secrets.compare_digest(provided, k) for k in valid)


async def require_api_key(key: str = Security(_header)) -> str:
    """FastAPI dependency — inject into routes that need auth """
    if not _auth_enabled():
        return "auth-disabled"

    if key is None:
        raise HTTPException(status_code=401, detail="Missing API key")

    valid = _valid_keys()
    if not valid:
        # No keys configured in env — refuse everything, do not silently allow
        raise HTTPException(status_code=500, detail="Server auth not configured")

    if not _is_valid_key(key, valid):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return key
