"""Tests for the per-service HTTP circuit breaker.

When a service (e.g. DBLP) returns sustained 429/5xx — IP-level throttling under
large-bibliography load — the client opens that service's circuit, stops
hammering it, emits a loud re-run warning, and persists the cooldown so a
too-soon re-run also self-paces.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bibtex_updater.utils import (
    CircuitOpenError,
    HttpClient,
    RateLimiterRegistry,
    SqliteCache,
)


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    # No real backoff sleeps, and no rate-limiter pacing (a no-op time.sleep would
    # otherwise make RateLimiter.wait busy-loop until real time advances).
    monkeypatch.setattr("bibtex_updater.utils.time.sleep", lambda *a, **k: None)
    monkeypatch.setattr("bibtex_updater.utils.RateLimiter.wait", lambda self: None)


def _client(cache=None, status=503):
    c = HttpClient(timeout=5.0, user_agent="test", rate_limiter=RateLimiterRegistry(), cache=cache)
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {}
    c.client = MagicMock()
    c.client.request.return_value = resp
    return c


def _fail_n(c, n, service="dblp"):
    for _ in range(n):
        with pytest.raises(RuntimeError):
            c._request("GET", "https://dblp.org/x", service=service)


class TestCircuitBreaker:
    def test_opens_after_threshold_then_skips_without_network(self):
        c = _client()
        _fail_n(c, HttpClient.CIRCUIT_FAIL_THRESHOLD)
        assert c.tripped_services == {"dblp"}
        c.client.request.reset_mock()
        with pytest.raises(CircuitOpenError):
            c._request("GET", "https://dblp.org/y", service="dblp")
        c.client.request.assert_not_called()  # skipped — no hammering

    def test_below_threshold_stays_closed(self):
        c = _client()
        _fail_n(c, HttpClient.CIRCUIT_FAIL_THRESHOLD - 1)
        assert "dblp" not in c.tripped_services
        assert not c._circuit_is_open("dblp")

    def test_success_resets_streak(self):
        c = _client()
        _fail_n(c, HttpClient.CIRCUIT_FAIL_THRESHOLD - 1)
        ok = MagicMock()
        ok.status_code = 200
        ok.headers = {"Content-Type": "text/plain"}
        c.client.request.return_value = ok
        c._request("GET", "https://dblp.org/ok", service="dblp")  # success resets streak
        bad = MagicMock()
        bad.status_code = 503
        bad.headers = {}
        c.client.request.return_value = bad
        _fail_n(c, HttpClient.CIRCUIT_FAIL_THRESHOLD - 1)
        assert "dblp" not in c.tripped_services  # streak reset, never reached threshold

    def test_other_services_unaffected(self):
        c = _client()
        _fail_n(c, HttpClient.CIRCUIT_FAIL_THRESHOLD)
        assert not c._circuit_is_open("crossref")
        c.client.request.reset_mock()
        with pytest.raises(RuntimeError):
            c._request("GET", "https://api.crossref.org/x", service="crossref")
        assert c.client.request.called  # crossref still hits the network

    def test_no_service_never_trips(self):
        c = _client()
        _fail_n(c, HttpClient.CIRCUIT_FAIL_THRESHOLD + 2, service=None)
        assert c.tripped_services == set()

    def test_persists_across_clients_via_cache(self, tmp_path):
        cache = SqliteCache(str(tmp_path / "c.db"))
        c1 = _client(cache=cache)
        _fail_n(c1, HttpClient.CIRCUIT_FAIL_THRESHOLD)
        assert c1.tripped_services == {"dblp"}
        # a fresh client sharing the cache inherits the still-open cooldown
        c2 = _client(cache=cache)
        with pytest.raises(CircuitOpenError):
            c2._request("GET", "https://dblp.org/z", service="dblp")
