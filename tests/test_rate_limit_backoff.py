"""Tests for rate-limit pacing fixes in ``bibtex-check``.

Two behaviours are covered, both targeting the user-reported timeout / rate-limit
symptoms while staying strictly verdict-neutral (pacing only — never which record
is fetched or how a verdict is computed):

P2 — ``build_rate_limits`` scales EVERY known service by ``--rate-limit`` (the old
inline dict scaled only 6 of ~9 services, so OpenAlex/arXiv ignored the knob).

P1 — the synchronous ``HttpClient._request`` retry loop now (a) honours a server
``Retry-After`` header on a 429 instead of blindly using its own exponential
backoff, and (b) feeds each response to ``AdaptiveRateLimiterRegistry.adapt`` so a
429 halves the per-service rate for subsequent requests (reducing 429 storms).

All tests are hermetic: the httpx transport is a canned stub; no network access.
"""

from __future__ import annotations

import httpx
import pytest

from bibtex_updater.fact_checker import build_rate_limits
from bibtex_updater.utils import (
    AdaptiveRateLimiterRegistry,
    HttpClient,
    RateLimiterRegistry,
)


class _ScriptedTransport:
    """httpx.Client stand-in that returns a scripted sequence of responses.

    Each ``request`` pops the next ``(status, headers)`` from the script; once
    exhausted it repeats the final entry. Bodies are an empty JSON object so the
    JSON cache path is exercised exactly like a real API response.
    """

    def __init__(self, script: list[tuple[int, dict[str, str]]]):
        self._script = script
        self.calls = 0

    def request(self, method, url, **kwargs):
        idx = min(self.calls, len(self._script) - 1)
        self.calls += 1
        status, extra_headers = self._script[idx]
        headers = {"Content-Type": "application/json", **extra_headers}
        return httpx.Response(
            status,
            content=b"{}",
            headers=headers,
            request=httpx.Request(method, str(url)),
        )


def _http(registry, transport) -> HttpClient:
    # No cache so retried statuses are never short-circuited by a cache hit;
    # high base limits so the rate limiter itself never sleeps in these tests.
    http = HttpClient(timeout=5.0, user_agent="test", rate_limiter=registry, cache=None)
    http.client = transport  # type: ignore[assignment]
    return http


# --------------------------------------------------------------------------- #
# P2: --rate-limit must scale ALL services, not just 6.
# --------------------------------------------------------------------------- #


def test_build_rate_limits_at_default_is_byte_identical_to_defaults():
    """At --rate-limit=45 (rate_scale=1.0) every service equals its default."""
    limits = build_rate_limits(rate_limit=45.0, s2_api_key=None)
    for service, default in RateLimiterRegistry.DEFAULT_LIMITS.items():
        if service == "semanticscholar":
            continue  # key-aware overlay handled separately
        assert limits[service] == default, service


def test_build_rate_limits_scales_openalex_and_arxiv():
    """The previously-unscaled polite-pool services now respond to --rate-limit.

    At half the default rate (22.5/45 = 0.5) OpenAlex (100) and arXiv (30) must
    halve; under the old inline dict they stayed pinned at their defaults.
    """
    limits = build_rate_limits(rate_limit=22.5, s2_api_key=None)
    assert limits["openalex"] == 50
    assert limits["arxiv"] == 15


def test_build_rate_limits_preserves_s2_key_aware_overlay():
    """Keyless S2 stays low (bot-detection mitigation); a key lifts it to 60."""
    keyless = build_rate_limits(rate_limit=45.0, s2_api_key=None)
    keyed = build_rate_limits(rate_limit=45.0, s2_api_key="secret")
    assert keyless["semanticscholar"] == 10
    assert keyed["semanticscholar"] == 60


def test_build_rate_limits_includes_book_verifier_services():
    """openlibrary / google_books are not in DEFAULT_LIMITS but must be present."""
    limits = build_rate_limits(rate_limit=45.0, s2_api_key=None)
    assert "openlibrary" in limits
    assert "google_books" in limits


# --------------------------------------------------------------------------- #
# P1: honour Retry-After + adaptive rate-halving in the sync retry loop.
# --------------------------------------------------------------------------- #


def test_request_honors_retry_after_header(monkeypatch):
    """A 429 carrying ``Retry-After: 3`` makes the retry sleep ~3s, not the
    exponential 1s the loop would otherwise use on the first retry."""
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    registry = RateLimiterRegistry(dict.fromkeys(RateLimiterRegistry.DEFAULT_LIMITS, 100_000))
    transport = _ScriptedTransport([(429, {"Retry-After": "3"}), (200, {})])
    http = _http(registry, transport)

    resp = http._request("GET", "https://api.crossref.org/works", service="crossref")

    assert resp.status_code == 200
    assert sleeps, "expected a backoff sleep between the 429 and the retry"
    # First (and only) backoff must honour the server's Retry-After of 3s,
    # not the loop's exponential first step of 1.0s.
    assert sleeps[0] == pytest.approx(3.0)


def test_request_caps_absurd_retry_after(monkeypatch):
    """A hostile/huge Retry-After is capped so one entry cannot stall for hours."""
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    registry = RateLimiterRegistry(dict.fromkeys(RateLimiterRegistry.DEFAULT_LIMITS, 100_000))
    transport = _ScriptedTransport([(429, {"Retry-After": "99999"}), (200, {})])
    http = _http(registry, transport)

    http._request("GET", "https://api.crossref.org/works", service="crossref")

    assert sleeps[0] <= 60.0


def test_adaptive_registry_halves_rate_on_429(monkeypatch):
    """A 429 must halve the per-service rate for subsequent requests via adapt().

    This proves adapt() is actually wired into the synchronous request path
    (previously it was dead code: only the async client reached it).
    """
    monkeypatch.setattr("time.sleep", lambda s: None)

    registry = AdaptiveRateLimiterRegistry({"crossref": 50})
    transport = _ScriptedTransport([(429, {}), (200, {})])
    http = _http(registry, transport)

    http._request("GET", "https://api.crossref.org/works", service="crossref")

    assert registry._limits["crossref"] == 25


def test_healthy_responses_do_not_change_rate(monkeypatch):
    """No 429 / no low-remaining header => the rate is untouched (perf-neutral)."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    registry = AdaptiveRateLimiterRegistry({"crossref": 50})
    transport = _ScriptedTransport([(200, {})])
    http = _http(registry, transport)

    http._request("GET", "https://api.crossref.org/works", service="crossref")

    assert registry._limits["crossref"] == 50
