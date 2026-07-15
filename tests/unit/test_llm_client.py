"""Tests for LLM client. All API calls mocked — no keys needed """

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from report_generation.llm_client import (
    LLMClient,
    LLMResponse,
    UsageStats,
    _backoff_delay,
    _cache_key,
    _format_anthropic,
    _format_openai,
    _parse_anthropic,
    _parse_openai,
)


# --- helpers ---

def _mock_anthropic_response(text="Report text.", in_tok=100, out_tok=50):
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def _mock_openai_response(text="Report text.", in_tok=100, out_tok=50):
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
    }


def _sample_prompt():
    return {"system": "You are a radiologist.", "user": "Findings: none.", "version": "v3"}


def _make_mock_response(status_code=200, json_data=None):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data or _mock_anthropic_response()
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        mock.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return mock


def _patch_httpx(mock_post):
    """Context manager that patches httpx.AsyncClient with a mock post method."""
    patcher = patch("httpx.AsyncClient")
    MockClient = patcher.start()
    instance = AsyncMock()
    instance.post = mock_post
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    MockClient.return_value = instance
    return patcher


# --- cache key ---

def test_cache_key_deterministic():
    k1 = _cache_key("system", "user")
    k2 = _cache_key("system", "user")
    assert k1 == k2


def test_cache_key_differs_for_different_prompts():
    k1 = _cache_key("system", "user A")
    k2 = _cache_key("system", "user B")
    assert k1 != k2


# --- backoff ---

def test_backoff_exponential():
    assert _backoff_delay(0) == 1.0
    assert _backoff_delay(1) == 2.0
    assert _backoff_delay(2) == 4.0


def test_backoff_capped():
    assert _backoff_delay(10) == 30.0


# --- provider formatting ---

def test_format_anthropic_structure():
    url, headers, body = _format_anthropic("sys", "usr", "claude-sonnet-4-6", 1024)
    assert "anthropic" in url
    assert "x-api-key" in headers
    assert body["model"] == "claude-sonnet-4-6"
    assert body["system"] == "sys"
    assert body["messages"][0]["content"] == "usr"


def test_format_openai_structure():
    url, headers, body = _format_openai("sys", "usr", "gpt-4o", 1024)
    assert "Authorization" in headers
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"


# --- response parsing ---

def test_parse_anthropic():
    data = _mock_anthropic_response("Hello.", 80, 20)
    text, in_t, out_t = _parse_anthropic(data)
    assert text == "Hello."
    assert in_t == 80
    assert out_t == 20


def test_parse_anthropic_empty():
    text, in_t, out_t = _parse_anthropic({})
    assert text == ""
    assert in_t == 0


def test_parse_openai():
    data = _mock_openai_response("Hello.", 80, 20)
    text, in_t, out_t = _parse_openai(data)
    assert text == "Hello."
    assert in_t == 80
    assert out_t == 20


def test_parse_openai_empty():
    text, in_t, out_t = _parse_openai({})
    assert text == ""


# --- LLMResponse ---

def test_response_cost_known_model():
    r = LLMResponse("text", 1000, 500, "claude-sonnet-4-6", 100.0)
    assert abs(r.estimated_cost_usd - 0.0105) < 1e-6


def test_response_cost_unknown_model():
    r = LLMResponse("text", 1000, 500, "some-new-model", 100.0)
    assert r.estimated_cost_usd == 0.0


# --- UsageStats ---

def test_stats_record():
    stats = UsageStats()
    r = LLMResponse("text", 100, 50, "claude-sonnet-4-6", 200.0)
    stats.record(r, retries=1)
    assert stats.total_calls == 1
    assert stats.total_input_tokens == 100
    assert stats.total_retries == 1
    assert stats.cache_hit_rate == 0.0


def test_stats_cache_hit_rate():
    stats = UsageStats()
    stats.record(LLMResponse("a", 100, 50, "m", 100.0, cached=False))
    stats.record(LLMResponse("a", 0, 0, "m", 0.0, cached=True))
    assert stats.cache_hit_rate == 0.5


# --- LLMClient init ---

def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        LLMClient(provider="gemini")


# --- LLMClient.generate (mocked, using asyncio.run) ---

def test_generate_success():
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-6")
    mock_post = AsyncMock(return_value=_make_mock_response(200, _mock_anthropic_response("The report.")))
    patcher = _patch_httpx(mock_post)

    try:
        result = asyncio.run(client.generate(_sample_prompt()))
    finally:
        patcher.stop()

    assert result.text == "The report."
    assert result.cached is False
    assert result.prompt_version == "v3"
    assert client.stats.total_calls == 1


def test_generate_cache_hit():
    client = LLMClient(provider="anthropic", cache_enabled=True)
    mock_post = AsyncMock(return_value=_make_mock_response(200, _mock_anthropic_response("Cached.")))
    patcher = _patch_httpx(mock_post)

    try:
        prompt = _sample_prompt()
        r1 = asyncio.run(client.generate(prompt))
        r2 = asyncio.run(client.generate(prompt))
    finally:
        patcher.stop()

    assert r1.cached is False
    assert r2.cached is True
    assert r2.text == "Cached."
    assert r2.latency_ms == 0.0


def test_generate_cache_disabled():
    client = LLMClient(provider="anthropic", cache_enabled=False)
    mock_post = AsyncMock(return_value=_make_mock_response(200, _mock_anthropic_response("Fresh.")))
    patcher = _patch_httpx(mock_post)

    try:
        prompt = _sample_prompt()
        asyncio.run(client.generate(prompt))
        asyncio.run(client.generate(prompt))
    finally:
        patcher.stop()

    assert mock_post.call_count == 2


def test_generate_retry_on_429():
    client = LLMClient(provider="anthropic", max_retries=2)
    fail_resp = _make_mock_response(429)
    ok_resp = _make_mock_response(200, _mock_anthropic_response("After retry."))
    mock_post = AsyncMock(side_effect=[fail_resp, ok_resp])
    patcher = _patch_httpx(mock_post)

    try:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(client.generate(_sample_prompt()))
    finally:
        patcher.stop()

    assert result.text == "After retry."
    assert client.stats.total_retries == 1


def test_generate_non_retryable_error():
    client = LLMClient(provider="anthropic", max_retries=2)
    mock_post = AsyncMock(return_value=_make_mock_response(401))
    patcher = _patch_httpx(mock_post)

    try:
        with pytest.raises(Exception, match="401"):
            asyncio.run(client.generate(_sample_prompt()))
    finally:
        patcher.stop()


def test_clear_cache():
    client = LLMClient()
    client._cache["abc"] = LLMResponse("x", 0, 0, "m", 0.0)
    assert client.clear_cache() == 1
    assert len(client._cache) == 0


# --- sync wrapper ---

def test_generate_sync():
    client = LLMClient(provider="anthropic")
    mock_post = AsyncMock(return_value=_make_mock_response(200, _mock_anthropic_response("Sync report.")))
    patcher = _patch_httpx(mock_post)

    try:
        result = client.generate_sync(_sample_prompt())
    finally:
        patcher.stop()

    assert result.text == "Sync report."