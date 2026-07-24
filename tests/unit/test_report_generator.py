"""End-to-end pipeline tests. LLM mocked, everything else real """

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from report_generation.feature_extractor import PATHOLOGIES
from report_generation.llm_client import LLMClient
from report_generation.report_generator import Report, ReportGenerator


# --- helpers ---

def _probs(*values) -> np.ndarray:
    """Build a probability vector, first N slots set to values."""
    arr = np.zeros(len(PATHOLOGIES))
    for i, v in enumerate(values):
        arr[i] = v
    return arr


def _mock_llm_response(text="FINDINGS: Consolidation in the right lower lobe."):
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 150, "output_tokens": 80},
    }


def _patched_client(response_text="FINDINGS: Consolidation noted.") -> tuple[LLMClient, MagicMock]:
    """Return an LLMClient with httpx patched to return the given response."""
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-6")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _mock_llm_response(response_text)
    mock_resp.raise_for_status = MagicMock()

    mock_post = AsyncMock(return_value=mock_resp)
    patcher = patch("httpx.AsyncClient")
    MockClient = patcher.start()

    instance = AsyncMock()
    instance.post = mock_post
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    MockClient.return_value = instance

    return client, patcher


# --- end-to-end pipeline ---

def test_generate_full_pipeline_grounded():
    client, patcher = _patched_client("Consolidation in the right lower lobe.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs(0.0, 0.0, 0.85)  # only Consolidation detected

    try:
        report = gen.generate_sync(probs, model_name="dinov2")
    finally:
        patcher.stop()

    assert isinstance(report, Report)
    assert "Consolidation" in report.text
    assert report.vision_model == "dinov2"
    assert report.llm_model == "claude-sonnet-4-6"
    assert report.prompt_version == "v3"


def test_generate_detects_hallucination():
    # LLM invents a pathology the vision model never flagged
    client, patcher = _patched_client("Consolidation and pneumothorax noted.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs(0.0, 0.0, 0.85)  # only Consolidation detected, no Pneumothorax

    try:
        report = gen.generate_sync(probs, model_name="dinov2")
    finally:
        patcher.stop()

    assert report.hallucination["is_clean"] is False
    assert "Pneumothorax" in report.hallucination["hallucinated"]
    assert "Consolidation" in report.hallucination["grounded"]


def test_generate_clean_report_no_hallucination():
    client, patcher = _patched_client("Consolidation in the right base. No effusion.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs(0.0, 0.0, 0.85)

    try:
        report = gen.generate_sync(probs, model_name="dinov2")
    finally:
        patcher.stop()

    assert report.hallucination["is_clean"] is True


def test_generate_appends_uncertainty_note_when_flagged():
    # low peak confidence + gray-zone finding → uncertainty flag
    client, patcher = _patched_client("Possible atelectasis, equivocal.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs(0.55, 0.35)

    try:
        report = gen.generate_sync(probs, model_name="dinov2")
    finally:
        patcher.stop()

    assert report.uncertainty["flag"] is True
    assert "UNCERTAINTY NOTICE" in report.text


def test_generate_no_uncertainty_note_when_confident():
    client, patcher = _patched_client("Consolidation in the right lower lobe.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs(0.0, 0.0, 0.90)  # confident single finding

    try:
        report = gen.generate_sync(probs, model_name="dinov2")
    finally:
        patcher.stop()

    assert report.uncertainty["flag"] is False
    assert "UNCERTAINTY NOTICE" not in report.text


def test_findings_dict_populated():
    client, patcher = _patched_client("Consolidation noted.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs(0.0, 0.0, 0.85)

    try:
        report = gen.generate_sync(probs, model_name="dinov2")
    finally:
        patcher.stop()

    assert "findings" in report.findings
    assert len(report.findings["findings"]) == 1
    assert report.findings["findings"][0]["pathology"] == "Consolidation"
    assert report.findings["model_name"] == "dinov2"


def test_provenance_populated():
    client, patcher = _patched_client("Report.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs(0.0, 0.0, 0.85)

    try:
        report = gen.generate_sync(probs, model_name="unet")
    finally:
        patcher.stop()

    assert report.input_tokens == 150
    assert report.output_tokens == 80
    assert report.cached is False


def test_second_call_cached():
    client, patcher = _patched_client("Cached report.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs(0.0, 0.0, 0.85)

    try:
        r1 = gen.generate_sync(probs, model_name="dinov2")
        r2 = gen.generate_sync(probs, model_name="dinov2")
    finally:
        patcher.stop()

    assert r1.cached is False
    assert r2.cached is True


def test_generate_with_segmentation_masks():
    client, patcher = _patched_client("Consolidation in the right lower lobe.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs(0.0, 0.0, 0.85)
    masks = np.zeros((len(PATHOLOGIES), 224, 224))
    masks[2, 160:200, 130:200] = 1.0

    try:
        report = gen.generate_sync(probs, segmentation_masks=masks, model_name="unet")
    finally:
        patcher.stop()

    assert report.findings["has_spatial_info"] is True
    assert report.findings["findings"][0]["location"] != "diffuse"


def test_empty_findings_normal_report():
    client, patcher = _patched_client("No acute cardiopulmonary findings.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs()  # nothing detected

    try:
        report = gen.generate_sync(probs, model_name="dinov2")
    finally:
        patcher.stop()

    assert report.findings["findings"] == []
    assert report.hallucination["is_clean"] is True


# --- serialization ---

def test_to_dict_shape():
    client, patcher = _patched_client("Report.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs(0.0, 0.0, 0.85)

    try:
        report = gen.generate_sync(probs, model_name="dinov2")
    finally:
        patcher.stop()

    d = report.to_dict()
    assert set(d.keys()) == {"text", "findings", "hallucination", "uncertainty", "provenance"}
    assert set(d["provenance"].keys()) == {
        "vision_model", "prompt_version", "llm_model",
        "latency_ms", "input_tokens", "output_tokens", "cached",
    }


# --- async entry point ---

def test_async_generate_works():
    client, patcher = _patched_client("Async report.")
    gen = ReportGenerator(llm_client=client)
    probs = _probs(0.0, 0.0, 0.85)

    try:
        report = asyncio.run(gen.generate(probs, model_name="dinov2"))
    finally:
        patcher.stop()

    assert "Async report" in report.text
