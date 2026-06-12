"""Tests for the Semantic Scholar ``/paper/batch`` bulk cache prefetch.

``FactCheckProcessor._batch_prefetch_s2_records`` POSTs all entry identifiers
(``DOI:<doi>`` / ``ARXIV:<id>``, deduped, DataCite arXiv DOIs skipped) to
``/paper/batch`` once and primes the HttpClient response cache with each
returned paper under the exact key ``SemanticScholarClient.get_paper(<id>)``
reads -- ``HttpClient.prime_cache`` shares its key construction with
``_request`` via a single private helper, so the two cannot drift.

Hermetic: a counting transport stands in for the network; the SqliteCache is
real so priming and cache hits are genuine.
"""

from __future__ import annotations

import json
import logging
import threading

import httpx
import pytest

from bibtex_updater.fact_checker import (
    CrossrefClient,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    FactCheckProcessor,
    SemanticScholarClient,
)
from bibtex_updater.utils import HttpClient, RateLimiterRegistry, SqliteCache

DOI = "10.1109/CVPR46437.2021.00469"
ARXIV = "2301.12345"

PAPER_DOI = {
    "title": "IBRNet: Learning Multi-View Image-Based Rendering",
    "venue": "CVPR",
    "year": 2021,
    "externalIds": {"DOI": DOI},
    "authors": [{"name": "Qianqian Wang"}],
}
PAPER_ARXIV = {
    "title": "Context-Aware Sparse Deep Coordination Graphs",
    "venue": "",
    "year": 2023,
    "externalIds": {"ArXiv": ARXIV},
    "authors": [{"name": "Tonghan Wang"}],
}


class _Transport:
    """Counting httpx.Client stand-in; routes by URL substring."""

    def __init__(self, batch_payload=None):
        self.batch_payload = batch_payload
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []
        self._lock = threading.Lock()

    def request(self, method, url, **kwargs):
        with self._lock:
            self.calls.append((method, str(url), kwargs.get("params"), kwargs.get("json")))
        if "paper/batch" in str(url):
            if isinstance(self.batch_payload, Exception):
                raise self.batch_payload
            return httpx.Response(
                200,
                content=json.dumps(self.batch_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                request=httpx.Request(method, url),
            )
        return httpx.Response(
            404,
            content=b"{}",
            headers={"Content-Type": "application/json"},
            request=httpx.Request(method, str(url)),
        )

    @property
    def count(self):
        return len(self.calls)


def _make(tmp_path, *, s2_api_key="real-key", cache=True, batch_payload=None):
    sqlite_cache = SqliteCache(str(tmp_path / "cache.db")) if cache else None
    registry = RateLimiterRegistry(dict.fromkeys(RateLimiterRegistry.DEFAULT_LIMITS, 100_000))
    http = HttpClient(
        timeout=5.0,
        user_agent="test",
        rate_limiter=registry,
        cache=sqlite_cache,  # type: ignore[arg-type]
        s2_api_key=s2_api_key,
    )
    transport = _Transport(batch_payload=batch_payload)
    http.client = transport  # type: ignore[assignment]
    checker = FactChecker(
        CrossrefClient(http),
        DBLPClient(http),
        SemanticScholarClient(http),
        FactCheckerConfig(),
        logging.getLogger("test_s2_batch"),
    )
    processor = FactCheckProcessor(checker, logging.getLogger("test_s2_batch"))
    return processor, checker, transport


def _doi_entry(entry_id="e1"):
    return {"ID": entry_id, "ENTRYTYPE": "inproceedings", "title": "T", "doi": DOI, "year": "2021"}


def _arxiv_entry(entry_id="e2"):
    return {
        "ID": entry_id,
        "ENTRYTYPE": "misc",
        "title": "T2",
        "eprint": ARXIV,
        "archiveprefix": "arXiv",
        "year": "2023",
    }


class TestBatchPrefetch:
    def test_one_batch_for_two_entries_sharing_ids_then_get_paper_is_cache_hit(self, tmp_path):
        processor, checker, transport = _make(tmp_path, batch_payload=[PAPER_DOI, PAPER_ARXIV])
        # Two entries sharing the same DOI plus one arXiv entry -> 2 deduped ids.
        entries = [_doi_entry("a"), _doi_entry("b"), _arxiv_entry("c")]

        primed = processor._batch_prefetch_s2_records(entries)

        assert primed == 2
        # Exactly ONE batch POST was issued.
        batch_calls = [c for c in transport.calls if "paper/batch" in c[1]]
        assert len(batch_calls) == 1
        method, _url, params, body = batch_calls[0]
        assert method == "POST"
        assert params == {"fields": SemanticScholarClient.PAPER_FIELDS}
        assert body == {"ids": [f"DOI:{DOI}", f"ARXIV:{ARXIV}"]}

        # The per-entry lookups are now pure cache hits: zero extra transport calls.
        before = transport.count
        assert checker.s2.get_paper(f"DOI:{DOI}") == PAPER_DOI
        assert checker.s2.get_paper(f"ARXIV:{ARXIV}") == PAPER_ARXIV
        assert transport.count == before

    def test_null_batch_members_are_skipped(self, tmp_path):
        processor, checker, transport = _make(tmp_path, batch_payload=[None, PAPER_ARXIV])
        entries = [_doi_entry(), _arxiv_entry()]

        primed = processor._batch_prefetch_s2_records(entries)

        assert primed == 1
        # The null id was NOT primed: its per-entry lookup goes to the network
        # (404 here) and returns None -- the unchanged fallback behavior.
        before = transport.count
        assert checker.s2.get_paper(f"DOI:{DOI}") is None
        assert transport.count == before + 1
        # The primed id is still a cache hit.
        before = transport.count
        assert checker.s2.get_paper(f"ARXIV:{ARXIV}") == PAPER_ARXIV
        assert transport.count == before

    def test_no_key_no_batch_call(self, tmp_path):
        processor, _checker, transport = _make(tmp_path, s2_api_key=None, batch_payload=[PAPER_DOI])
        assert processor._batch_prefetch_s2_records([_doi_entry()]) == 0
        assert transport.count == 0

    def test_blank_key_no_batch_call(self, tmp_path):
        processor, _checker, transport = _make(tmp_path, s2_api_key="  ", batch_payload=[PAPER_DOI])
        assert processor._batch_prefetch_s2_records([_doi_entry()]) == 0
        assert transport.count == 0

    def test_no_cache_no_batch_call(self, tmp_path):
        processor, _checker, transport = _make(tmp_path, cache=False, batch_payload=[PAPER_DOI])
        assert processor._batch_prefetch_s2_records([_doi_entry()]) == 0
        assert transport.count == 0

    def test_no_identifiers_no_batch_call(self, tmp_path):
        processor, _checker, transport = _make(tmp_path, batch_payload=[])
        entry = {"ID": "x", "ENTRYTYPE": "article", "title": "No ids here", "year": "2020"}
        assert processor._batch_prefetch_s2_records([entry]) == 0
        assert transport.count == 0

    def test_datacite_arxiv_doi_sent_as_arxiv_id_not_doi(self, tmp_path):
        processor, _checker, transport = _make(tmp_path, batch_payload=[PAPER_ARXIV])
        entry = {
            "ID": "d",
            "ENTRYTYPE": "misc",
            "title": "T3",
            "doi": f"10.48550/arXiv.{ARXIV}",
            "year": "2023",
        }
        primed = processor._batch_prefetch_s2_records([entry])

        assert primed == 1
        batch_calls = [c for c in transport.calls if "paper/batch" in c[1]]
        assert len(batch_calls) == 1
        ids = batch_calls[0][3]["ids"]
        assert ids == [f"ARXIV:{ARXIV}"]
        assert not any(i.startswith("DOI:") for i in ids)

    def test_chunking_at_500_ids(self, tmp_path):
        n = 501
        payload = [PAPER_DOI] * 500  # each chunk gets a full answer list
        processor, _checker, transport = _make(tmp_path, batch_payload=payload)
        entries = [
            {"ID": f"e{i}", "ENTRYTYPE": "article", "title": "T", "doi": f"10.1234/x{i}", "year": "2020"}
            for i in range(n)
        ]
        processor._batch_prefetch_s2_records(entries)

        batch_calls = [c for c in transport.calls if "paper/batch" in c[1]]
        assert len(batch_calls) == 2
        assert len(batch_calls[0][3]["ids"]) == 500
        assert len(batch_calls[1][3]["ids"]) == 1

    def test_transport_error_is_swallowed_and_per_entry_falls_back(self, tmp_path, caplog):
        # A non-httpx error propagates straight out of _request (no retry
        # loop), which keeps this hermetic test instant.
        processor, checker, transport = _make(tmp_path, batch_payload=RuntimeError("boom"))
        with caplog.at_level(logging.DEBUG, logger="test_s2_batch"):
            primed = processor._batch_prefetch_s2_records([_doi_entry()])
        assert primed == 0
        # Best-effort: the failure is debug-logged, never raised.
        assert any("best-effort" in r.getMessage() for r in caplog.records)
        # Per-entry get_paper still performs its own fetch.
        before = transport.count
        assert checker.s2.get_paper(f"DOI:{DOI}") is None  # 404 route
        assert transport.count == before + 1

    def test_non_200_batch_response_skipped(self, tmp_path):
        # 400 (non-retryable, non-200) -> chunk skipped, nothing primed.
        class _Status400Transport(_Transport):
            def request(self, method, url, **kwargs):
                resp = super().request(method, url, **kwargs)
                if "paper/batch" in str(url):
                    return httpx.Response(
                        400,
                        content=b"{}",
                        headers={"Content-Type": "application/json"},
                        request=httpx.Request(method, str(url)),
                    )
                return resp

        processor, checker, _ = _make(tmp_path, batch_payload=[PAPER_DOI])
        transport = _Status400Transport(batch_payload=[PAPER_DOI])
        checker.crossref.http.client = transport  # type: ignore[assignment]
        assert processor._batch_prefetch_s2_records([_doi_entry()]) == 0

    def test_duck_typed_checker_is_a_noop(self, tmp_path):
        class _Duck:
            def check_entry(self, entry):  # pragma: no cover - never called here
                raise AssertionError

        processor = FactCheckProcessor(_Duck(), logging.getLogger("test_s2_batch"))  # type: ignore[arg-type]
        assert processor._batch_prefetch_s2_records([_doi_entry()]) == 0


class TestPrimeCacheKeyParity:
    """prime_cache and _request share one key builder; prove the parity."""

    def test_primed_value_served_by_request(self, tmp_path):
        _processor, checker, transport = _make(tmp_path, batch_payload=[])
        http = checker.crossref.http
        url = "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1/abc"
        params = {"fields": "title"}

        http.prime_cache("GET", url, params=params, accept="application/json", value={"title": "Primed"})
        resp = http._request("GET", url, params=params, accept="application/json", service="semanticscholar")

        assert resp.status_code == 200
        assert resp.headers.get("X-From-Cache") == "1"
        assert resp.json() == {"title": "Primed"}
        assert transport.count == 0

    def test_different_params_do_not_collide(self, tmp_path):
        _processor, checker, transport = _make(tmp_path, batch_payload=[])
        http = checker.crossref.http
        url = "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1/abc"

        http.prime_cache("GET", url, params={"fields": "title"}, accept="application/json", value={"t": 1})
        # A request with DIFFERENT params must miss the primed entry.
        resp = http._request("GET", url, params={"fields": "title,venue"}, accept="application/json")
        assert resp.headers.get("X-From-Cache") is None
        assert transport.count == 1

    def test_prime_cache_noop_without_cache(self, tmp_path):
        registry = RateLimiterRegistry(dict.fromkeys(RateLimiterRegistry.DEFAULT_LIMITS, 100_000))
        http = HttpClient(timeout=5.0, user_agent="test", rate_limiter=registry, cache=None)  # type: ignore[arg-type]
        # Must not raise.
        http.prime_cache("GET", "https://x.test/", params=None, accept=None, value={"a": 1})


class TestProcessEntriesIntegration:
    def test_process_entries_primes_then_preprint_check_hits_cache(self, tmp_path):
        """End-to-end: process_entries issues ONE batch POST and the per-entry
        S2 get_paper lookups (preprint check) add ZERO further S2 /paper GETs."""
        processor, _checker, transport = _make(tmp_path, batch_payload=[PAPER_DOI])
        entries = [_doi_entry("a")]
        processor.process_entries(entries, max_workers=1)

        batch_calls = [c for c in transport.calls if "paper/batch" in c[1]]
        assert len(batch_calls) == 1
        single_paper_gets = [c for c in transport.calls if "/paper/DOI:" in c[1] or "/paper/ARXIV:" in c[1]]
        assert single_paper_gets == []

    def test_process_entries_without_key_issues_no_batch(self, tmp_path):
        processor, _checker, transport = _make(tmp_path, s2_api_key=None, batch_payload=[PAPER_DOI])
        processor.process_entries([_doi_entry("a")], max_workers=1)
        assert [c for c in transport.calls if "paper/batch" in c[1]] == []


@pytest.mark.parametrize("payload", [{"unexpected": "dict"}, "junk", 42])
def test_non_list_batch_response_is_ignored(tmp_path, payload):
    processor, checker, transport = _make(tmp_path, batch_payload=payload)
    assert processor._batch_prefetch_s2_records([_doi_entry()]) == 0
    # Nothing primed -> the per-entry lookup still goes out.
    before = transport.count
    checker.s2.get_paper(f"DOI:{DOI}")
    assert transport.count == before + 1
