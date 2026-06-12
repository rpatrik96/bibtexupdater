"""Tests for the CLI rate-limit profile, adaptive-registry wiring, and polite
mailto identity.

Covers:
  * ``_cli_service_rate_limits``: scaled-and-capped per-service limits
    (Crossref/OpenAlex raised toward documented ceilings, arXiv lowered to the
    politeness ask, S2 keyed vs keyless).
  * ``RateLimiterRegistry.DEFAULT_LIMITS``: arXiv 30 -> 20.
  * ``HttpClient._request`` feeding every REAL response (including retryable
    429s) to ``rate_limiter.adapt`` while cache hits never adapt.
  * ``AdaptiveRateLimiterRegistry``: 429 + Retry-After sets a backoff that
    ``wait`` honors (with the ``_backoff_until`` access now lock-guarded).
  * ``--mailto`` flag / ``BIBTEX_CHECK_MAILTO`` env plumbing into the
    User-Agent and the OpenAlex mailto default, and the one-time warning when
    no contact address is configured.

All fakes / hermetic transports; behavior proven by counting calls on fakes.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import httpx
import pytest

import bibtex_updater.fact_checker as fact_checker_module
from bibtex_updater.fact_checker import (
    DEFAULT_MAILTO_PLACEHOLDER,
    _cli_service_rate_limits,
    _effective_openalex_mailto,
    _polite_user_agent,
    _resolve_polite_mailto,
    build_parser,
)
from bibtex_updater.sources import DEFAULT_OPENALEX_MAILTO
from bibtex_updater.utils import (
    AdaptiveRateLimiterRegistry,
    HttpClient,
    RateLimiterRegistry,
    SqliteCache,
)

# ===========================================================================
# Per-service CLI rate limits
# ===========================================================================


class TestCliServiceRateLimits:
    def test_scale_one_defaults(self):
        limits = _cli_service_rate_limits(45, None)
        assert limits == {
            "crossref": 300,
            "openalex": 150,
            "dblp": 30,
            "openreview": 30,
            "arxiv": 20,
            "semanticscholar": 10,
            "openlibrary": 30,
            "google_books": 30,
        }

    def test_scale_one_with_key_s2_is_60(self):
        limits = _cli_service_rate_limits(45, "some-key")
        assert limits["semanticscholar"] == 60
        # The key changes nothing else.
        assert limits["crossref"] == 300
        assert limits["arxiv"] == 20

    def test_scale_two_hits_caps(self):
        limits = _cli_service_rate_limits(90, None)
        assert limits["crossref"] == 600
        assert limits["openalex"] == 300
        assert limits["dblp"] == 60
        assert limits["openreview"] == 60
        # arXiv NEVER scales up: ~1 req/3s politeness ask.
        assert limits["arxiv"] == 20

    def test_huge_scale_still_capped(self):
        limits = _cli_service_rate_limits(900, "key")
        assert limits["crossref"] == 600
        assert limits["openalex"] == 300
        assert limits["dblp"] == 60
        assert limits["openreview"] == 60
        assert limits["arxiv"] == 20
        assert limits["semanticscholar"] == 60

    def test_tiny_scale_hits_floors(self):
        limits = _cli_service_rate_limits(1, None)
        assert limits["crossref"] == 10
        assert limits["openalex"] == 10
        assert limits["dblp"] == 10
        assert limits["openreview"] == 10
        assert limits["arxiv"] == 20
        assert limits["semanticscholar"] == 5
        assert limits["openlibrary"] == 10
        assert limits["google_books"] == 10

    def test_registry_default_arxiv_lowered_to_politeness_ask(self):
        assert RateLimiterRegistry.DEFAULT_LIMITS["arxiv"] == 20


# ===========================================================================
# HttpClient -> adapt() wiring
# ===========================================================================


class _RecordingRegistry(RateLimiterRegistry):
    """Registry that records adapt() calls (high limits: never sleeps)."""

    def __init__(self):
        super().__init__(dict.fromkeys(RateLimiterRegistry.DEFAULT_LIMITS, 100_000))
        self.adapt_calls: list[tuple[str, int]] = []

    def adapt(self, service, response):
        self.adapt_calls.append((service, response.status_code))


class _ScriptedTransport:
    """httpx.Client stand-in returning scripted responses; counts real calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.count = 0
        self._lock = threading.Lock()

    def request(self, method, url, **kwargs):
        with self._lock:
            self.count += 1
            spec = self._responses.pop(0) if self._responses else (200, {})
        status, headers = spec
        return httpx.Response(
            status,
            content=json.dumps({"ok": True}).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            request=httpx.Request(method, url),
        )


def _http(registry, tmp_path, responses):
    cache = SqliteCache(str(tmp_path / "cache.db"))
    http = HttpClient(timeout=5.0, user_agent="test", rate_limiter=registry, cache=cache)
    transport = _ScriptedTransport(responses)
    http.client = transport  # type: ignore[assignment]
    return http, transport


class TestAdaptWiring:
    def test_adapt_called_once_per_real_response_and_not_on_cache_hit(self, tmp_path):
        registry = _RecordingRegistry()
        http, transport = _http(registry, tmp_path, [(200, {})])

        resp = http._request("GET", "https://api.crossref.org/works", params={"q": "x"}, service="crossref")
        assert resp.status_code == 200
        assert transport.count == 1
        assert registry.adapt_calls == [("crossref", 200)]

        # Identical request -> served from cache -> NO new adapt call.
        resp2 = http._request("GET", "https://api.crossref.org/works", params={"q": "x"}, service="crossref")
        assert resp2.headers.get("X-From-Cache") == "1"
        assert transport.count == 1
        assert registry.adapt_calls == [("crossref", 200)]

    def test_adapt_sees_retryable_429_before_retry(self, tmp_path):
        registry = _RecordingRegistry()
        # First a 429 with an instant Retry-After (keeps the test fast), then 200.
        http, transport = _http(registry, tmp_path, [(429, {"Retry-After": "0"}), (200, {})])

        resp = http._request("GET", "https://api.crossref.org/works", params={"q": "y"}, service="crossref")
        assert resp.status_code == 200
        assert transport.count == 2
        # BOTH transport responses were fed to adapt, 429 first.
        assert registry.adapt_calls == [("crossref", 429), ("crossref", 200)]

    def test_no_service_no_adapt(self, tmp_path):
        registry = _RecordingRegistry()
        http, _ = _http(registry, tmp_path, [(200, {})])
        http._request("GET", "https://example.org/a")
        assert registry.adapt_calls == []

    def test_plain_registry_without_adapt_is_fine(self, tmp_path):
        registry = RateLimiterRegistry(dict.fromkeys(RateLimiterRegistry.DEFAULT_LIMITS, 100_000))
        http, transport = _http(registry, tmp_path, [(200, {})])
        resp = http._request("GET", "https://example.org/b", service="crossref")
        assert resp.status_code == 200
        assert transport.count == 1

    def test_adapt_exception_never_loses_the_response(self, tmp_path):
        class _ExplodingRegistry(_RecordingRegistry):
            def adapt(self, service, response):
                super().adapt(service, response)
                raise RuntimeError("adaptation bug")

        registry = _ExplodingRegistry()
        http, _ = _http(registry, tmp_path, [(200, {})])
        resp = http._request("GET", "https://example.org/c", service="crossref")
        assert resp.status_code == 200
        assert registry.adapt_calls == [("crossref", 200)]


# ===========================================================================
# AdaptiveRateLimiterRegistry: backoff behavior
# ===========================================================================


def _resp_429(headers=None):
    return httpx.Response(429, headers=headers or {}, request=httpx.Request("GET", "https://x.test/"))


class TestAdaptiveBackoff:
    def test_429_with_retry_after_sets_backoff(self):
        reg = AdaptiveRateLimiterRegistry({"crossref": 1000})
        before = time.time()
        reg.adapt("crossref", _resp_429({"Retry-After": "30"}))
        with reg._lock:
            backoff_end = reg._backoff_until["crossref"]
        assert backoff_end == pytest.approx(before + 30.0, abs=1.0)
        # And the limit was halved.
        assert reg._limits["crossref"] == 500

    def test_429_without_retry_after_defaults_to_60s(self):
        reg = AdaptiveRateLimiterRegistry({"crossref": 1000})
        before = time.time()
        reg.adapt("crossref", _resp_429())
        with reg._lock:
            backoff_end = reg._backoff_until["crossref"]
        assert backoff_end == pytest.approx(before + 60.0, abs=1.0)

    def test_wait_honors_backoff(self, monkeypatch):
        reg = AdaptiveRateLimiterRegistry({"crossref": 100_000})
        reg.adapt("crossref", _resp_429({"Retry-After": "5"}))

        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        reg.wait("crossref")
        assert slept and slept[0] == pytest.approx(5.0, abs=1.0)

    def test_wait_without_backoff_does_not_sleep(self, monkeypatch):
        reg = AdaptiveRateLimiterRegistry({"crossref": 100_000})
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        reg.wait("crossref")
        assert slept == []

    def test_backoff_state_is_lock_guarded(self):
        # Hammer adapt() and wait() concurrently; with the lock in place this
        # must be free of raised exceptions and end with a sane backoff value.
        reg = AdaptiveRateLimiterRegistry({"crossref": 100_000})
        errors: list[BaseException] = []

        def _adapt():
            try:
                for _ in range(200):
                    reg.adapt("crossref", _resp_429({"Retry-After": "0"}))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def _wait():
            try:
                for _ in range(200):
                    reg.wait("crossref")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_adapt), threading.Thread(target=_wait)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        with reg._lock:
            assert "crossref" in reg._backoff_until


# ===========================================================================
# Polite mailto identity
# ===========================================================================


@pytest.fixture
def _fresh_mailto_state(monkeypatch):
    monkeypatch.delenv("BIBTEX_CHECK_MAILTO", raising=False)
    monkeypatch.setattr(fact_checker_module, "_mailto_warning_emitted", False)


class TestPoliteMailto:
    def test_parser_accepts_mailto_flag(self):
        args = build_parser().parse_args(["refs.bib", "--mailto", "alice@lab.edu"])
        assert args.mailto == "alice@lab.edu"

    def test_parser_default_is_none(self):
        args = build_parser().parse_args(["refs.bib"])
        assert args.mailto is None

    def test_flag_wins_over_env(self, _fresh_mailto_state, monkeypatch):
        monkeypatch.setenv("BIBTEX_CHECK_MAILTO", "env@lab.edu")
        logger = logging.getLogger("test_mailto")
        assert _resolve_polite_mailto("flag@lab.edu", logger) == "flag@lab.edu"

    def test_env_used_when_flag_absent(self, _fresh_mailto_state, monkeypatch):
        monkeypatch.setenv("BIBTEX_CHECK_MAILTO", "env@lab.edu")
        logger = logging.getLogger("test_mailto")
        assert _resolve_polite_mailto(None, logger) == "env@lab.edu"

    def test_warning_emitted_once_when_unset(self, _fresh_mailto_state, caplog):
        logger = logging.getLogger("test_mailto_warn")
        with caplog.at_level(logging.WARNING, logger="test_mailto_warn"):
            assert _resolve_polite_mailto(None, logger) is None
            assert _resolve_polite_mailto(None, logger) is None
        warnings = [r for r in caplog.records if "No contact email configured" in r.getMessage()]
        assert len(warnings) == 1
        assert "BIBTEX_CHECK_MAILTO" in warnings[0].getMessage()

    def test_no_warning_when_set(self, _fresh_mailto_state, caplog):
        logger = logging.getLogger("test_mailto_warn2")
        with caplog.at_level(logging.WARNING, logger="test_mailto_warn2"):
            assert _resolve_polite_mailto("a@b.c", logger) == "a@b.c"
        assert [r for r in caplog.records if "No contact email" in r.getMessage()] == []

    def test_user_agent_with_real_mailto(self):
        assert _polite_user_agent("alice@lab.edu") == "BibtexFactChecker/1.0 (mailto:alice@lab.edu)"

    def test_user_agent_placeholder_unchanged_when_unset(self):
        # The historical default UA must be preserved byte-for-byte.
        assert _polite_user_agent(None) == f"BibtexFactChecker/1.0 (mailto:{DEFAULT_MAILTO_PLACEHOLDER})"
        assert _polite_user_agent(None) == "BibtexFactChecker/1.0 (mailto:factchecker@example.com)"

    def test_openalex_mailto_defaults_to_global_mailto(self):
        assert _effective_openalex_mailto(DEFAULT_OPENALEX_MAILTO, "alice@lab.edu") == "alice@lab.edu"

    def test_explicit_openalex_mailto_wins(self):
        assert _effective_openalex_mailto("oa@lab.edu", "alice@lab.edu") == "oa@lab.edu"

    def test_openalex_default_kept_when_no_mailto(self):
        assert _effective_openalex_mailto(DEFAULT_OPENALEX_MAILTO, None) == DEFAULT_OPENALEX_MAILTO
