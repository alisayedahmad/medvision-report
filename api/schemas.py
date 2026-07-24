"""API request and response schemas.

These define the contract between the API and its clients.
Everything the API accepts or returns is shaped here.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class FindingResponse(BaseModel):
    """Single detected pathology with location and confidence."""

    pathology: str = Field(..., description="Detected pathology name")
    location: str = Field(..., description="Anatomical location")
    bbox: list[int] = Field(
        default_factory=list,
        description="Bounding box [x, y, width, height]",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: str = Field(..., description="low / moderate / high")


class AnalysisResponse(BaseModel):
    """Full report returned by POST /analyze."""

    findings: list[FindingResponse] = Field(default_factory=list)
    report: str = Field(..., description="LLM-generated clinical report")
    uncertainty_flag: bool = Field(
        default=False,
        description="True when model confidence is below threshold",
    )
    processing_time_ms: float = Field(..., ge=0.0)
    model_version: str = Field(default="v1.0.0")


class HealthResponse(BaseModel):
    """Liveness and readiness info for K8s probes."""

    status: str = Field(..., description="ok / degraded / unavailable")
    model_loaded: bool = False
    uptime_seconds: float = Field(..., ge=0.0)
    version: str = Field(default="v1.0.0")


class ErrorResponse(BaseModel):
    """Consistent error format so clients can parse failures."""

    status_code: int
    detail: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
