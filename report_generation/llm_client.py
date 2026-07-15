"""Async LLM client for clinical report generation.

Handles retry with backoff, prompt-level caching, and token/cost tracking.
Uses httpx directly instead of SDK — no provider lock-in, fully testable.
Supports Anthropic Claude (default) and OpenAI-compatible APIs.

"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# retry on these status codes
_RETRYABLE = {429, 500, 502, 503, 529}

# rough cost per 1M tokens (USD) — update as pricing changes
_COST_PER_1M: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.0},
}


@dataclass
class LLMResponse:
    """Everything downstream needs from a single LLM call."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    cached: bool = False
    prompt_version: str = ""

    @property
    def estimated_cost_usd(self) -> float:
        rates = _COST_PER_1M.get(self.model, {"input": 0, "output": 0})
        return (
            self.input_tokens * rates["input"]
            + self.output_tokens * rates["output"]
        ) / 1_000_000


@dataclass
class UsageStats:
    """Cumulative stats across multiple calls. Reset per session """

    total_calls: int = 0
    cached_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    total_retries: int = 0

    def record(self, response: LLMResponse, retries: int = 0) -> None:
        self.total_calls += 1
        if response.cached:
            self.cached_calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_latency_ms += response.latency_ms
        self.total_retries += retries

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_calls / self.total_calls if self.total_calls else 0.0


def _cache_key(system: str, user: str) -> str:
    """SHA256 of the full prompt. Deterministic, no collisions in practice."""
    content = f"{system}\n---\n{user}"
    return hashlib.sha256(content.encode()).hexdigest()


def _format_anthropic(
    system: str, user: str, model: str, max_tokens: int
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build Anthropic API request."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    return url, headers, body


def _format_openai(
    system: str, user: str, model: str, max_tokens: int
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build OpenAI-compatible API request."""
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    url = f"{base}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    return url, headers, body


def _parse_anthropic(data: dict) -> tuple[str, int, int]:
    """Extract text and token counts from Anthropic response."""
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    usage = data.get("usage", {})
    return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def _parse_openai(data: dict) -> tuple[str, int, int]:
    """Extract text and token counts from OpenAI response."""
    choices = data.get("choices", [])
    text = choices[0]["message"]["content"] if choices else ""
    usage = data.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


_PROVIDERS = {
    "anthropic": {"format": _format_anthropic, "parse": _parse_anthropic},
    "openai": {"format": _format_openai, "parse": _parse_openai},
}


class LLMClient:
    """Async LLM client with retry, caching, and usage tracking.

    Usage:
        client = LLMClient(provider="anthropic", model="claude-sonnet-4-6")
        prompt = build_prompt(findings_dict)  # from prompts.py
        response = await client.generate(prompt)
        # or synchronously:
        response = client.generate_sync(prompt)
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        max_retries: int = 3,
        timeout: float = 30.0,
        cache_enabled: bool = True,
    ):
        if provider not in _PROVIDERS:
            raise ValueError(f"unknown provider '{provider}', use: {list(_PROVIDERS)}")

        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_enabled = cache_enabled

        self._formatter = _PROVIDERS[provider]["format"]
        self._parser = _PROVIDERS[provider]["parse"]
        self._cache: dict[str, LLMResponse] = {}
        self.stats = UsageStats()

    async def generate(self, prompt: dict[str, str]) -> LLMResponse:
        """Send prompt to LLM, return structured response.

        Args:
            prompt: {"system": ..., "user": ..., "version": ...} from prompts.py
        """
        system = prompt["system"]
        user = prompt["user"]
        version = prompt.get("version", "")

        # cache check
        key = _cache_key(system, user)
        if self.cache_enabled and key in self._cache:
            cached = self._cache[key]
            response = LLMResponse(
                text=cached.text,
                input_tokens=0,  # no API call made
                output_tokens=0,
                model=cached.model,
                latency_ms=0.0,
                cached=True,
                prompt_version=version,
            )
            self.stats.record(response)
            return response

        # build request
        url, headers, body = self._formatter(system, user, self.model, self.max_tokens)

        # call with retry
        text, in_tokens, out_tokens, latency, retries = await self._call_with_retry(
            url, headers, body
        )

        response = LLMResponse(
            text=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model=self.model,
            latency_ms=latency,
            cached=False,
            prompt_version=version,
        )

        if self.cache_enabled:
            self._cache[key] = response

        self.stats.record(response, retries=retries)
        return response

    def generate_sync(self, prompt: dict[str, str]) -> LLMResponse:
        """Sync wrapper — for notebooks and testing."""
        return asyncio.run(self.generate(prompt))

    async def _call_with_retry(
        self, url: str, headers: dict, body: dict
    ) -> tuple[str, int, int, float, int]:
        """POST with exponential backoff on transient errors."""
        retries = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                t0 = time.monotonic()
                try:
                    resp = await client.post(url, headers=headers, json=body)
                    latency = (time.monotonic() - t0) * 1000

                    if resp.status_code == 200:
                        data = resp.json()
                        text, in_tok, out_tok = self._parser(data)
                        return text, in_tok, out_tok, latency, retries

                    if resp.status_code in _RETRYABLE and attempt < self.max_retries:
                        wait = _backoff_delay(attempt)
                        logger.warning(
                            "LLM API %d, retry %d/%d in %.1fs",
                            resp.status_code, attempt + 1, self.max_retries, wait,
                        )
                        retries += 1
                        await asyncio.sleep(wait)
                        continue

                    # non-retryable error or out of retries
                    resp.raise_for_status()

                except httpx.TimeoutException:
                    latency = (time.monotonic() - t0) * 1000
                    if attempt < self.max_retries:
                        wait = _backoff_delay(attempt)
                        logger.warning(
                            "LLM API timeout, retry %d/%d in %.1fs",
                            attempt + 1, self.max_retries, wait,
                        )
                        retries += 1
                        await asyncio.sleep(wait)
                        continue
                    raise

        # should never reach here, but just in case
        raise RuntimeError("LLM API call failed after all retries")

    def clear_cache(self) -> int:
        """Clear prompt cache, return number of entries removed."""
        n = len(self._cache)
        self._cache.clear()
        return n


def _backoff_delay(attempt: int, base: float = 1.0, max_delay: float = 30.0) -> float:
    """Exponential backoff: 1s, 2s, 4s, ... capped at max_delay."""
    return min(base * (2 ** attempt), max_delay)
