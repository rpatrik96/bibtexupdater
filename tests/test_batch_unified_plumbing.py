"""Batch DOI optimizations must reach the CLI path (UnifiedFactChecker).

The CLI (``main()``) always wraps the academic checker in a
:class:`UnifiedFactChecker`, but ``FactCheckProcessor`` used to gate its two
batch optimizations on ``isinstance(self.checker, FactChecker)``:

  * ``_process_one`` only forwarded ``pre_validated_dois`` to a bare
    FactChecker, so the batch doi.org HEAD-sweep results were computed then
    DISCARDED and every DOI entry re-HEADed doi.org serially (uncached);
  * ``_batch_warm_crossref_records`` returned 0 immediately, so the Crossref
    ``/works`` cache warm-up never ran in production.

These tests drive a FactCheckProcessor wrapping a UnifiedFactChecker built on
fake clients (a counting transport stub; no network) and prove:
  (a) after ``process_entries`` the doi.org HEAD for each DOI happened exactly
      once -- the per-entry validation consumed the batch result;
  (b) ``_batch_warm_crossref_records`` returns >0 and the per-entry
      ``get_by_doi`` is then served from the cache (one underlying fetch per
      distinct DOI).
"""

from __future__ import annotations

import collections
import json
import logging
import threading
from unittest.mock import MagicMock
from urllib.parse import unquote

import httpx
import pytest

from bibtex_updater.fact_checker import (
    CrossrefClient,
    DBLPClient,
    EntryCategory,
    FactChecker,
    FactCheckerConfig,
    FactCheckProcessor,
    FactCheckStatus,
    SemanticScholarClient,
    UnifiedFactChecker,
)
from bibtex_updater.utils import HttpClient, RateLimiterRegistry, SqliteCache


def _works_payload(title: str, doi: str) -> dict:
    """A Crossref ``/works/{doi}`` REST response (``message`` wraps one record)."""
    return {
        "message": {
            "DOI": doi,
            "type": "proceedings-article",
            "title": [title],
            "author": [{"given": "A", "family": "Author"}],
            "issued": {"date-parts": [[2021]]},
        }
    }


class _StubTransport:
    """Thread-safe ``httpx.Client`` stand-in counting uncached round-trips.

    * ``HEAD https://doi.org/<doi>`` -> 200, counted per DOI (the batch sweep
      and any per-entry ``_validate_doi`` fallback both land here);
    * ``GET api.crossref.org/works/<doi>`` -> canned 200 payload, counted per
      DOI (cache misses only -- the SqliteCache in ``HttpClient._request``
      eliminates duplicates);
    * everything else (search endpoints of every source) -> empty 200 JSON.
    """

    def __init__(self, works: dict[str, dict]):
        self._works = {k.lower(): v for k, v in works.items()}
        self.doi_head_counts: collections.Counter[str] = collections.Counter()
        self.works_get_counts: collections.Counter[str] = collections.Counter()
        self._lock = threading.Lock()

    def head(self, url, **kwargs):
        url_str = str(url)
        assert "doi.org" in url_str, f"unexpected HEAD to {url_str}"
        doi = url_str.lower().split("doi.org/", 1)[1]
        with self._lock:
            self.doi_head_counts[doi] += 1
        return httpx.Response(200, request=httpx.Request("HEAD", url_str))

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def request(self, method, url, **kwargs):
        url_str = str(url).lower()
        request = httpx.Request(method, url_str)
        if "api.crossref.org/works/" in url_str:
            doi = unquote(url_str.split("/works/", 1)[1])
            with self._lock:
                self.works_get_counts[doi] += 1
            payload = self._works.get(doi)
            if payload is None:
                return httpx.Response(404, content=b"{}", headers={"Content-Type": "application/json"}, request=request)
            return httpx.Response(
                200,
                content=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                request=request,
            )
        # Search endpoints of every cascade source: valid-but-empty JSON.
        return httpx.Response(200, content=b"{}", headers={"Content-Type": "application/json"}, request=request)


def _make_unified(works: dict[str, dict], tmp_path, logger, with_cache: bool = True):
    """UnifiedFactChecker on a real HttpClient + SqliteCache, stub transport."""
    cache = SqliteCache(str(tmp_path / "cache.db")) if with_cache else None
    registry = RateLimiterRegistry(dict.fromkeys(RateLimiterRegistry.DEFAULT_LIMITS, 100_000))
    http = HttpClient(timeout=5.0, user_agent="test", rate_limiter=registry, cache=cache)
    http.client.close()  # replace the real client with the stub
    transport = _StubTransport(works)
    http.client = transport  # type: ignore[assignment]
    unified = UnifiedFactChecker(
        http=http,
        crossref=CrossrefClient(http),
        dblp=DBLPClient(http),
        s2=SemanticScholarClient(http),
        config=FactCheckerConfig(),
        logger=logger,
    )
    return unified, transport


def _entries() -> list[dict[str, str]]:
    # Wrong DOIs (resolve to different papers) -> deterministic DOI_MISMATCH,
    # which exercises both the HEAD sweep and the /works consistency fetch.
    return [
        {
            "ID": "ibrnet2021",
            "ENTRYTYPE": "inproceedings",
            "title": "IBRNet: Learning Multi-View Image-Based Rendering",
            "author": "Wang, Qianqian and Wang, Zhicheng",
            "doi": "10.1109/CVPR46437.2021.00469",
            "year": "2021",
        },
        {
            "ID": "imagebind2023",
            "ENTRYTYPE": "inproceedings",
            "title": "ImageBind: One Embedding Space To Bind Them All",
            "author": "Girdhar, Rohit and El-Nouby, Alaaeldin",
            "doi": "10.1109/CVPR52729.2023.01457",
            "year": "2023",
        },
    ]


def _works_routes() -> dict[str, dict]:
    return {
        "10.1109/cvpr46437.2021.00469": _works_payload(
            "Delving into Localization Errors for Monocular 3D Object Detection",
            "10.1109/CVPR46437.2021.00469",
        ),
        "10.1109/cvpr52729.2023.01457": _works_payload(
            "Segment Anything in High Quality",
            "10.1109/CVPR52729.2023.01457",
        ),
    }


@pytest.fixture
def logger():
    return logging.getLogger("test_batch_unified_plumbing")


class TestBatchDoiSweepReachesUnified:
    """(a) The batch HEAD-sweep result is CONSUMED by per-entry validation."""

    def test_one_head_per_doi_through_process_entries(self, logger, tmp_path, monkeypatch):
        unified, transport = _make_unified(_works_routes(), tmp_path, logger)
        processor = FactCheckProcessor(unified, logger)

        # The batch sweep must route through the SHARED client (the stub), not
        # a throwaway httpx.Client whose real-network HEADs silently fail (the
        # pre-fix UnifiedFactChecker behavior).
        def _no_throwaway_client(*args, **kwargs):
            raise AssertionError("throwaway httpx.Client constructed on the batch path")

        monkeypatch.setattr(httpx, "Client", _no_throwaway_client)

        results = processor.process_entries(_entries(), max_workers=2)

        # The pipeline genuinely ran: both wrong DOIs were flagged.
        assert [r.status for r in results] == [
            FactCheckStatus.DOI_MISMATCH,
            FactCheckStatus.DOI_MISMATCH,
        ]
        # Exactly ONE doi.org HEAD per DOI: the batch sweep. The per-entry
        # _validate_doi consumed the pre-validated result instead of
        # re-HEADing doi.org serially (the pre-fix behavior was 2 per DOI).
        assert dict(transport.doi_head_counts) == {
            "10.1109/cvpr46437.2021.00469": 1,
            "10.1109/cvpr52729.2023.01457": 1,
        }

    def test_unified_check_entry_consumes_pre_validated_result(self, logger, tmp_path):
        """A pre-validated False short-circuits to DOI_NOT_FOUND with no HEAD."""
        unified, transport = _make_unified(_works_routes(), tmp_path, logger)
        entry = _entries()[0]

        result = unified.check_entry(entry, pre_validated_dois={entry["ID"]: False})

        assert result.status == FactCheckStatus.DOI_NOT_FOUND
        assert transport.doi_head_counts.total() == 0


class TestCrossrefWarmupReachesUnified:
    """(b) The /works cache warm-up runs for UnifiedFactChecker and the
    per-entry get_by_doi calls become cache hits."""

    def test_warm_returns_count_and_per_entry_hits_cache(self, logger, tmp_path):
        unified, transport = _make_unified(_works_routes(), tmp_path, logger)
        processor = FactCheckProcessor(unified, logger)
        entries = _entries()

        warmed = processor._batch_warm_crossref_records(entries)

        assert warmed == 2  # pre-fix: returned 0 for UnifiedFactChecker
        assert transport.works_get_counts.total() == 2  # one fetch per DOI

        # Per-entry get_by_doi (the consistency-check fetch) is a cache hit:
        # zero additional underlying fetches.
        crossref = unified.academic_fact_checker.crossref
        for entry in entries:
            assert crossref.get_by_doi(entry["doi"]) is not None
        assert transport.works_get_counts.total() == 2

    def test_full_run_fetches_each_works_record_once(self, logger, tmp_path):
        """End-to-end: warm-up + per-entry consistency check = ONE fetch/DOI."""
        unified, transport = _make_unified(_works_routes(), tmp_path, logger)
        processor = FactCheckProcessor(unified, logger)

        processor.process_entries(_entries(), max_workers=2)

        assert dict(transport.works_get_counts) == {
            "10.1109/cvpr46437.2021.00469": 1,
            "10.1109/cvpr52729.2023.01457": 1,
        }

    def test_warm_noop_without_cache(self, logger, tmp_path):
        """No shared response cache -> warming stays a documented no-op."""
        unified, transport = _make_unified(_works_routes(), tmp_path, logger, with_cache=False)
        processor = FactCheckProcessor(unified, logger)

        assert processor._batch_warm_crossref_records(_entries()) == 0
        assert transport.works_get_counts.total() == 0


class TestAcademicCheckerAccessor:
    """The shared-client accessor resolves through either wrapper type."""

    def test_unified_exposes_inner_academic_checker(self, logger, tmp_path):
        unified, _ = _make_unified({}, tmp_path, logger)

        inner = unified.academic_fact_checker

        assert isinstance(inner, FactChecker)
        verifier = unified.verifiers[EntryCategory.ACADEMIC]
        assert inner is verifier.fact_checker  # type: ignore[attr-defined]

    def test_processor_resolves_both_wrappers(self, logger, tmp_path):
        unified, _ = _make_unified({}, tmp_path, logger)
        assert FactCheckProcessor(unified, logger)._academic_checker() is unified.academic_fact_checker

        bare = unified.academic_fact_checker
        assert FactCheckProcessor(bare, logger)._academic_checker() is bare

        # Duck-typed stand-ins (tests) expose no shared clients.
        assert FactCheckProcessor(MagicMock(spec=[]), logger)._academic_checker() is None
