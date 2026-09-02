"""A source that never answered is not evidence that a reference is absent.

``not_found`` is an EXHAUSTIVE claim: every source consulted for the entry
completed its lookup, and none holds a matching record. Before this contract was
enforced, every source client swallowed its failures and returned an empty list,
so a lookup that never left the machine was indistinguishable downstream from a
database that had checked and found nothing.

The incident (2026-09-02): mid-run wifi loss during a 5,043-reference check.
Every subsequent lookup failed DNS resolution, every entry came back
``not_found``, nothing warned, exit code 0 -- 2,500 real, correctly-cited
references reported as unknown to every database. In a citation-hallucination
screen ``not_found`` is the status that feeds "this reference may be fabricated",
so this is the worst direction for the failure to point.

All network access is faked; no test here touches a live host.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import httpx
import pytest

from bibtex_updater.fact_checker import (
    EXIT_SOURCE_OUTAGE,
    NETWORK_OUTAGE_ENTRY_FRACTION,
    ArxivClient,
    CrossrefClient,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    FactCheckProcessor,
    FactCheckResult,
    FactCheckStatus,
    SemanticScholarClient,
    _report_source_outage,
)
from bibtex_updater.sources import OpenAlexClient, OpenReviewClient
from bibtex_updater.utils import (
    HttpClient,
    RateLimiterRegistry,
    SourceUnavailableError,
)

LOGGER = logging.getLogger("test_source_outage")

#: What macOS reports when DNS resolution fails after the wifi drops. httpx
#: surfaces the underlying ``socket.gaierror`` as ``ConnectError``.
DNS_FAILURE = "[Errno 8] nodename nor servname provided, or not known"


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """No retry backoff and no rate-limiter pacing (mirrors test_circuit_breaker)."""
    monkeypatch.setattr("bibtex_updater.utils.time.sleep", lambda *a, **k: None)
    monkeypatch.setattr("bibtex_updater.utils.RateLimiter.wait", lambda self: None)


def _entry() -> dict[str, str]:
    """A real, correctly-cited paper with no DOI and no arXiv ID, so the whole
    verdict rests on the title/author cascade."""
    return {
        "ID": "widget2023",
        "ENTRYTYPE": "article",
        "title": "Sparse Coding for Widget Recognition",
        "author": "Smith, Jane and Doe, John",
        "journal": "Journal of Widget Research",
        "year": "2023",
    }


def _http(side_effect=None, response=None) -> HttpClient:
    """A real HttpClient (retry loop, circuit breaker) over a faked transport."""
    http = HttpClient(timeout=1.0, user_agent="test", rate_limiter=RateLimiterRegistry(), cache=None)
    http.client = MagicMock()
    if side_effect is not None:
        http.client.request.side_effect = side_effect
    else:
        http.client.request.return_value = response
    return http


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


def _checker(http: HttpClient) -> FactChecker:
    """A FactChecker on the real client adapters, so failures travel the whole
    path from the transport to the emitted status."""
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


# ===========================================================================
# The transport layer: a failed lookup is raised, not swallowed
# ===========================================================================


class TestHttpClientSurfacesFailedLookups:
    def test_dns_failure_raises_and_names_the_host(self):
        http = _http(side_effect=httpx.ConnectError(DNS_FAILURE))
        with pytest.raises(SourceUnavailableError) as exc_info:
            http._request("GET", "https://api.crossref.org/works", service="crossref")
        exc = exc_info.value
        assert exc.service == "crossref"
        assert exc.host == "api.crossref.org"
        assert exc.transport_failure is True
        assert DNS_FAILURE in str(exc)
        assert http.unreachable_hosts == {"api.crossref.org": 1}

    def test_429_is_a_failed_lookup_but_not_an_unreachable_host(self):
        """An HTTP error response proves DNS resolved, TCP connected and TLS
        negotiated. The lookup still failed, but the network is demonstrably up,
        so a throttled source must not be reported as an outage."""
        resp = MagicMock()
        resp.status_code = 429
        resp.headers = {}
        http = _http(response=resp)
        with pytest.raises(SourceUnavailableError) as exc_info:
            http._request("GET", "https://api.crossref.org/works", service="crossref")
        assert exc_info.value.transport_failure is False
        assert http.unreachable_hosts == {}
        assert http.failed_services == {"crossref": 1}

    def test_5xx_after_retries_counts_as_unreachable(self):
        resp = MagicMock()
        resp.status_code = 503
        resp.headers = {}
        http = _http(response=resp)
        with pytest.raises(SourceUnavailableError) as exc_info:
            http._request("GET", "https://dblp.org/search/publ/api", service="dblp")
        assert exc_info.value.transport_failure is True
        assert http.unreachable_hosts == {"dblp.org": 1}

    def test_failure_is_still_a_runtime_error(self):
        """Callers that caught the bare RuntimeError the retry loop used to
        raise keep working."""
        http = _http(side_effect=httpx.ConnectError(DNS_FAILURE))
        with pytest.raises(RuntimeError):
            http._request("GET", "https://api.crossref.org/works", service="crossref")


# ===========================================================================
# The verdict: not_found requires a technically successful check
# ===========================================================================


class TestNotFoundRequiresCompleteCoverage:
    def test_total_dns_failure_is_not_reported_as_not_found(self):
        """The incident, end to end: the wifi is gone, no source can be reached,
        and the entry must not come back as a reference no database knows."""
        http = _http(side_effect=httpx.ConnectError(DNS_FAILURE))
        result = _checker(http).check_entry(_entry())

        assert result.status is not FactCheckStatus.NOT_FOUND
        assert result.status is FactCheckStatus.API_ERROR
        assert result.sources_failed  # every source that was asked
        assert result.coverage_incomplete is True
        assert http.unreachable_hosts  # named for the run-level report

    def test_genuine_empty_answer_is_still_not_found(self):
        """The control: every source answered, none holds the paper. That IS an
        exhaustive miss and must keep reporting one -- the fix must not turn the
        checker into something that can never say not_found."""
        http = _http(side_effect=_EmptyTransport())
        result = _checker(http).check_entry(_entry())

        assert result.status is FactCheckStatus.NOT_FOUND
        assert result.sources_failed == []
        assert result.errors == []
        assert result.coverage_incomplete is False

    def test_one_failed_source_blocks_the_exhaustive_claim(self):
        """A partial cascade cannot support an exhaustive miss, even when every
        other source answered cleanly and found nothing."""
        empty = _EmptyTransport()

        def transport(method, url, **kwargs):
            if "dblp.org" in url:
                raise httpx.ConnectError(DNS_FAILURE)
            return empty()

        http = _http(side_effect=transport)
        result = _checker(http).check_entry(_entry())

        assert result.status is not FactCheckStatus.NOT_FOUND
        assert result.status is FactCheckStatus.API_ERROR
        assert result.sources_failed == ["dblp"]
        assert result.coverage_incomplete is True

    def test_failed_source_names_reach_the_jsonl(self):
        http = _http(side_effect=httpx.ConnectError(DNS_FAILURE))
        result = _checker(http).check_entry(_entry())
        processor = FactCheckProcessor(MagicMock(), LOGGER)
        record = json.loads(processor.generate_jsonl([result])[0])

        assert record["status"] == "api_error"
        assert record["coverage_incomplete"] is True
        assert record["sources_failed"] == result.sources_failed


# ===========================================================================
# The run: an outage is loud and does not exit 0
# ===========================================================================


def _result(status: FactCheckStatus, sources_failed: list[str] | None = None) -> FactCheckResult:
    return FactCheckResult(
        entry_key="k",
        entry_type="article",
        status=status,
        overall_confidence=0.0,
        field_comparisons={},
        best_match=None,
        api_sources_queried=["crossref"],
        api_sources_with_hits=[],
        errors=[f"Crossref: {s} lookup did not complete" for s in sources_failed or []],
        sources_failed=list(sources_failed or []),
    )


class TestRunLevelOutageReport:
    def test_summary_counts_entries_and_sources(self):
        results = [
            _result(FactCheckStatus.API_ERROR, ["crossref", "dblp"]),
            _result(FactCheckStatus.API_ERROR, ["crossref"]),
            _result(FactCheckStatus.VERIFIED),
        ]
        summary = FactCheckProcessor(MagicMock(), LOGGER).generate_summary(results)

        assert summary["sources_failed_count"] == 2
        assert summary["failed_source_counts"] == {"crossref": 2, "dblp": 1}

    def test_poisoned_run_warns_names_the_hosts_and_exits_nonzero(self, caplog):
        summary = {"total": 100, "sources_failed_count": 90, "failed_source_counts": {"crossref": 90}}
        http = MagicMock()
        http.unreachable_hosts = {"api.crossref.org": 90}

        with caplog.at_level(logging.WARNING, logger=LOGGER.name):
            code = _report_source_outage(summary, http, LOGGER)

        assert code == EXIT_SOURCE_OUTAGE != 0
        text = caplog.text
        assert "api.crossref.org" in text
        assert "not_found" in text

    def test_healthy_run_exits_zero_and_says_nothing(self, caplog):
        summary = {"total": 100, "sources_failed_count": 0, "failed_source_counts": {}}
        with caplog.at_level(logging.WARNING, logger=LOGGER.name):
            assert _report_source_outage(summary, MagicMock(), LOGGER) == 0
        assert caplog.text == ""

    def test_isolated_failures_are_reported_without_condemning_the_run(self, caplog):
        """Below the threshold the affected entries are still unusable and still
        logged; the rest of the run stands."""
        failed = int(NETWORK_OUTAGE_ENTRY_FRACTION * 100) - 1
        summary = {"total": 100, "sources_failed_count": failed, "failed_source_counts": {"dblp": failed}}
        http = MagicMock()
        http.unreachable_hosts = {}

        with caplog.at_level(logging.WARNING, logger=LOGGER.name):
            assert _report_source_outage(summary, http, LOGGER) == 0
        assert "dblp" in caplog.text
