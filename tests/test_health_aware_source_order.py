"""A source that is down should not be the first one every entry waits on.

The cascade consults its metadata sources in a fixed order, and it short-circuits
as soon as one of them fully confirms the entry, so an early source is consulted
for nearly every reference. When that source is unreachable, every entry pays its
retry budget and its timeout before a healthy source gets a turn.

The measurement behind this (2026-09-02, a two-day screening run over 5,043
references): a reachability probe sampled four sources every five minutes and
found dblp unreachable in 32 of 36 samples, sustained across both days, returning
connection failures rather than HTTP errors. Crossref, OpenAlex and arXiv stayed
healthy throughout. The flat 90-second cooldown then had the run re-attempting
dblp roughly 40 times an hour, four failures each, so it never settled into
skipping a service that was down for two days.

What this does NOT claim: dblp was not the dominant cause of the throughput drop
observed on that run. That was self-inflicted rate limiting, from three processes
each running its own limiter against healthy sources. Health-aware ordering
stands on its own.

Two invariants carry the risk, and both are pinned below. Ordering must be
identical to the legacy cascade while every source is healthy, and reordering
must never change the SET of sources consulted, because ``not_found`` is the
exhaustive claim that every source answered (the v1.8.0 contract).

All network access is faked; no test here touches a live host.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import httpx
import pytest

from bibtex_updater.fact_checker import (
    ArxivClient,
    CrossrefClient,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    FactCheckStatus,
    SemanticScholarClient,
)
from bibtex_updater.sources import OpenAlexClient, OpenReviewClient
from bibtex_updater.utils import HttpClient, RateLimiterRegistry

LOGGER = logging.getLogger("test_health_aware_source_order")

#: What macOS reports when DNS resolution fails after the wifi drops.
DNS_FAILURE = "[Errno 8] nodename nor servname provided, or not known"

#: The cascade order that runs when every source answers. Health-aware ordering
#: is a stable sort on the health tier alone, so this list must survive it
#: unchanged. The two ``-fallback`` names are the relaxed-author retry, which
#: runs after the cascade when nothing usable was found.
DECLARED_ORDER = [
    "crossref",
    "openalex",
    "dblp",
    "openreview",
    "semanticscholar",
    "crossref-fallback",
    "openalex-fallback",
]


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """No retry backoff and no rate-limiter pacing (mirrors test_circuit_breaker)."""
    monkeypatch.setattr("bibtex_updater.utils.time.sleep", lambda *a, **k: None)
    monkeypatch.setattr("bibtex_updater.utils.RateLimiter.wait", lambda self: None)


def _entry() -> dict[str, str]:
    """A DOI-less, arXiv-less paper, so the whole verdict rests on the cascade."""
    return {
        "ID": "widget2023",
        "ENTRYTYPE": "article",
        "title": "Sparse Coding for Widget Recognition",
        "author": "Smith, Jane and Doe, John",
        "journal": "Journal of Widget Research",
        "year": "2023",
    }


def _ok(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"Content-Type": "application/json"}
    resp.json.return_value = payload
    return resp


class _EmptyTransport:
    """Every endpoint answers 200 with its own shape of "nothing found"."""

    EMPTY = {
        "message": {"items": []},  # Crossref
        "results": [],  # OpenAlex
        "result": {"hits": {}},  # DBLP
        "data": [],  # Semantic Scholar
        "notes": [],  # OpenReview
    }

    def __call__(self, *args, **kwargs) -> MagicMock:
        return _ok(dict(self.EMPTY))


def _http(side_effect=None, s2_api_key: str | None = None) -> HttpClient:
    """A real HttpClient (retry loop, circuit breaker) over a faked transport."""
    http = HttpClient(
        timeout=1.0,
        user_agent="test",
        rate_limiter=RateLimiterRegistry(),
        cache=None,
        s2_api_key=s2_api_key,
    )
    http.client = MagicMock()
    http.client.request.side_effect = side_effect if side_effect is not None else _EmptyTransport()
    return http


def _checker(http: HttpClient) -> FactChecker:
    """A FactChecker on the real client adapters, so health travels the whole
    path from the circuit breaker to the order the cascade runs in."""
    return FactChecker(
        CrossrefClient(http),
        DBLPClient(http),
        SemanticScholarClient(http),
        FactCheckerConfig(check_dois=False, check_doi_consistency=False, check_arxiv_consistency=False),
        LOGGER,
        openalex=OpenAlexClient(http=http),
        arxiv=ArxivClient(http),
        openreview=OpenReviewClient(http=http),
    )


def _open_circuit(http: HttpClient, service: str, seconds: float = 300.0) -> None:
    """Open ``service``'s circuit directly, the same state ``_record_service_failure``
    writes after a sustained outage."""
    http._circuit_open_until[service] = time.time() + seconds


def _trip(http: HttpClient, service: str = "dblp") -> None:
    """Drive the breaker the way a real sustained outage does."""
    for _ in range(HttpClient.CIRCUIT_FAIL_THRESHOLD):
        http._record_service_failure(service)


# ===========================================================================
# The health signal: read off the existing circuit state, not a second tally
# ===========================================================================


class TestServiceHealth:
    def test_an_untouched_service_is_healthy(self):
        http = _http()
        assert http.service_health("dblp") == HttpClient.HEALTH_OK
        assert http.open_circuits == set()

    def test_an_open_circuit_is_the_unhealthiest_tier(self):
        http = _http()
        _trip(http, "dblp")
        assert http.service_health("dblp") == HttpClient.HEALTH_CIRCUIT_OPEN
        assert http.open_circuits == {"dblp"}
        assert http.service_health("crossref") == HttpClient.HEALTH_OK

    def test_consistent_failures_degrade_before_the_breaker_trips(self):
        """A source failing repeatedly is demoted before it trips, so the run
        stops leading with it a few entries earlier."""
        http = _http()
        http._record_service_failure("dblp")
        assert http.service_health("dblp") == HttpClient.HEALTH_OK  # one blip is not a signal
        for _ in range(HttpClient.CIRCUIT_DEGRADED_STREAK - 1):
            http._record_service_failure("dblp")
        assert http.service_health("dblp") == HttpClient.HEALTH_DEGRADED
        assert http.open_circuits == set()  # degraded, not tripped

    def test_a_success_restores_health(self):
        http = _http()
        for _ in range(HttpClient.CIRCUIT_DEGRADED_STREAK):
            http._record_service_failure("dblp")
        http._record_service_success("dblp")
        assert http.service_health("dblp") == HttpClient.HEALTH_OK

    def test_an_expired_cooldown_leaves_open_circuits(self):
        http = _http()
        _trip(http, "dblp")
        http._circuit_open_until["dblp"] = time.time() - 1.0
        assert http.open_circuits == set()
        assert http.service_health("dblp") != HttpClient.HEALTH_CIRCUIT_OPEN
        assert http.tripped_services == {"dblp"}  # sticky for the run-level report

    def test_an_unnamed_service_is_never_demoted(self):
        assert _http().service_health(None) == HttpClient.HEALTH_OK

    def test_the_tiers_sort_healthiest_first(self):
        assert HttpClient.HEALTH_OK < HttpClient.HEALTH_DEGRADED < HttpClient.HEALTH_CIRCUIT_OPEN


# ===========================================================================
# Backing off a service that stays dead
# ===========================================================================


class TestCooldownEscalation:
    def _reopen(self, http: HttpClient, times: int, service: str = "dblp") -> list[float]:
        """Trip the circuit ``times`` times, expiring the cooldown in between,
        and return the cooldown each opening asked for."""
        cooldowns = []
        for _ in range(times):
            _trip(http, service)
            cooldowns.append(http._circuit_open_until[service] - time.time())
            http._circuit_open_until[service] = 0.0
            http._circuit_fail_streak[service] = 0
        return cooldowns

    def test_the_first_cooldown_stays_short(self):
        """A brief blip must recover fast, so escalation starts from the
        unchanged 90 seconds."""
        http = _http()
        assert self._reopen(http, 1) == [pytest.approx(HttpClient.CIRCUIT_COOLDOWN, abs=1.0)]

    def test_the_cooldown_grows_on_each_reopening(self):
        http = _http()
        cooldowns = self._reopen(http, 4)
        assert cooldowns == [
            pytest.approx(90.0, abs=1.0),
            pytest.approx(180.0, abs=1.0),
            pytest.approx(360.0, abs=1.0),
            pytest.approx(720.0, abs=1.0),
        ]

    def test_the_cooldown_is_capped(self):
        http = _http()
        cooldowns = self._reopen(http, 12)
        assert cooldowns[-1] == pytest.approx(HttpClient.CIRCUIT_COOLDOWN_MAX, abs=1.0)
        assert max(cooldowns) <= HttpClient.CIRCUIT_COOLDOWN_MAX + 1.0

    def test_escalation_is_per_service(self):
        http = _http()
        self._reopen(http, 3, "dblp")
        assert self._reopen(http, 1, "crossref") == [pytest.approx(HttpClient.CIRCUIT_COOLDOWN, abs=1.0)]

    def test_a_dead_service_settles_into_far_fewer_probes_per_hour(self):
        """The throughput claim, in one assertion. At the flat cooldown a service
        that is down for an hour is retried ~40 times; escalation brings that to
        a handful."""
        http = _http()
        elapsed = 0.0
        probes = 0
        while elapsed < 3600.0:
            _trip(http)
            elapsed += http._circuit_open_until["dblp"] - time.time()
            probes += 1
            http._circuit_open_until["dblp"] = 0.0
            http._circuit_fail_streak["dblp"] = 0
        assert probes <= 6
        assert 3600.0 / HttpClient.CIRCUIT_COOLDOWN > 35  # what the flat cooldown did


# ===========================================================================
# The ordering itself
# ===========================================================================


class TestHealthOrderedSteps:
    def test_all_healthy_is_the_identity(self):
        fc = _checker(_http())
        steps = [(name, lambda: None) for name in ("crossref", "openalex", "dblp", "openreview")]
        assert [name for name, _ in fc._health_ordered_steps(steps)] == [name for name, _ in steps]

    def test_two_steps_of_one_source_keep_their_relative_order(self):
        """The Semantic Scholar match and search steps share a source name, so a
        stable sort on the health tier can never separate or swap them."""
        http = _http()
        _open_circuit(http, "semanticscholar")
        fc = _checker(http)
        steps = [
            ("semanticscholar", lambda: None),
            ("crossref", lambda: None),
            ("semanticscholar", lambda: None),
        ]
        ordered = fc._health_ordered_steps(steps)
        assert [name for name, _ in ordered] == ["crossref", "semanticscholar", "semanticscholar"]
        assert [fn for _, fn in ordered] == [steps[1][1], steps[0][1], steps[2][1]]

    def test_degraded_sorts_between_healthy_and_open(self):
        http = _http()
        _open_circuit(http, "dblp")
        for _ in range(HttpClient.CIRCUIT_DEGRADED_STREAK):
            http._record_service_failure("openalex")
        fc = _checker(http)
        steps = [(name, lambda: None) for name in ("dblp", "openalex", "crossref")]
        assert [name for name, _ in fc._health_ordered_steps(steps)] == ["crossref", "openalex", "dblp"]

    def test_a_client_without_a_health_probe_keeps_the_declared_order(self):
        """Mocked and third-party clients report no health. Absence of a signal
        must never reorder anything."""
        fc = _checker(_http())
        fc.crossref = MagicMock()  # a Mock answers service_health with a Mock
        steps = [(name, lambda: None) for name in ("dblp", "crossref", "openalex")]
        assert [name for name, _ in fc._health_ordered_steps(steps)] == ["dblp", "crossref", "openalex"]


class TestCascadeOrder:
    def test_all_healthy_runs_the_declared_order(self):
        """The regression guard: with every source answering, the cascade is
        byte-for-byte what it was before health-aware ordering existed."""
        result = _checker(_http()).check_entry(_entry())
        assert result.api_sources_queried == DECLARED_ORDER

    def test_an_open_circuit_moves_a_source_last_and_still_consults_it(self):
        http = _http()
        _open_circuit(http, "dblp")
        result = _checker(http).check_entry(_entry())

        assert result.api_sources_queried.index("dblp") > result.api_sources_queried.index("openreview")
        assert result.api_sources_queried.index("dblp") > result.api_sources_queried.index("semanticscholar")
        assert "dblp" in result.api_sources_queried  # demoted, never dropped
        assert set(result.api_sources_queried) == set(DECLARED_ORDER)

    def test_the_semantic_scholar_match_step_moves_with_its_search_step(self):
        """With an API key the cascade gains a second Semantic Scholar step. Both
        carry the same source name, so demoting the source moves the pair and
        keeps the match step ahead of the search step."""
        healthy = _checker(_http(s2_api_key="k")).check_entry(_entry()).api_sources_queried
        assert healthy.index("semanticscholar") == 1  # right after Crossref

        http = _http(s2_api_key="k")
        _open_circuit(http, "semanticscholar")
        demoted = _checker(http).check_entry(_entry()).api_sources_queried
        assert demoted.index("semanticscholar") > demoted.index("openreview")


class TestReorderingPreservesTheVerdict:
    """``not_found`` means every source answered (the v1.8.0 contract). Ordering
    may change which source answers first; it may never change who was asked."""

    def _dblp_is_down(self):
        empty = _EmptyTransport()

        def transport(method, url, **kwargs):
            if "dblp.org" in url:
                raise httpx.ConnectError(DNS_FAILURE)
            return empty()

        return transport

    def test_status_and_sources_failed_are_invariant_under_reordering(self):
        """Same source outcomes, two different orders, one verdict. dblp fails in
        both runs: in the first it is consulted in its declared slot, in the
        second its open circuit has demoted it to the end."""
        declared = _checker(_http(side_effect=self._dblp_is_down())).check_entry(_entry())

        http = _http(side_effect=self._dblp_is_down())
        _open_circuit(http, "dblp")
        reordered = _checker(http).check_entry(_entry())

        assert declared.api_sources_queried != reordered.api_sources_queried  # the order did change
        assert declared.status is reordered.status is FactCheckStatus.API_ERROR
        assert set(declared.sources_failed) == set(reordered.sources_failed) == {"dblp"}
        assert declared.coverage_incomplete is reordered.coverage_incomplete is True

    def test_a_clean_miss_is_still_not_found_when_a_source_was_demoted(self):
        """The control. Demoting a source that then answers cleanly must not cost
        the run its ability to say not_found."""
        http = _http()
        for _ in range(HttpClient.CIRCUIT_DEGRADED_STREAK):
            http._record_service_failure("dblp")  # degraded, so demoted, but reachable
        result = _checker(http).check_entry(_entry())

        assert result.status is FactCheckStatus.NOT_FOUND
        assert result.sources_failed == []
        assert result.coverage_incomplete is False
        assert set(result.api_sources_queried) == set(DECLARED_ORDER)
