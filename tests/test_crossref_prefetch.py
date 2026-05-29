"""Tests for the up-front Crossref ``/works`` cache-warming in ``bibtex-check``.

Every entry that carries a DOI triggers per-entry Crossref ``/works/{doi}``
fetches (``_check_doi_consistency`` and the structured-name author recheck, both
via ``CrossrefClient.get_by_doi``). Run serially across the worker pool, those
round-trips are gated by the crossref rate limiter (50/min) and dominate
wall-clock on large bibliographies.

``FactCheckProcessor._batch_warm_crossref_records`` pre-fetches all entry DOIs
ONCE, up front, in parallel, calling the SAME ``get_by_doi`` path so the shared
``HttpClient`` SqliteCache is warmed. The per-entry calls then become cache hits.

These tests prove:
  (a) warming serves the records -- the underlying HTTP fetch count drops to one
      per distinct DOI (the warm), and per-entry ``get_by_doi`` calls are hits;
  (b) verdict-neutrality -- statuses are IDENTICAL with and without warming;
  (c) a pre-fetch failure for one DOI still yields the correct per-entry result
      (graceful fallback to the existing per-entry fetch, no behavior change).

All tests are hermetic: the httpx transport is replaced by a counting stub; no
network access.
"""

from __future__ import annotations

import json
import logging
import threading
from unittest.mock import MagicMock

import httpx
import pytest

from bibtex_updater.fact_checker import (
    CrossrefClient,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    FactCheckProcessor,
    FactCheckStatus,
    SemanticScholarClient,
)
from bibtex_updater.utils import HttpClient, RateLimiterRegistry, SqliteCache


def _works_payload(title: str, doi: str, authors: list[dict] | None = None) -> dict:
    """A Crossref ``/works/{doi}`` REST response (``message`` wraps one record)."""
    return {
        "message": {
            "DOI": doi,
            "type": "proceedings-article",
            "title": [title],
            "author": authors or [{"given": "A", "family": "Author"}],
            "issued": {"date-parts": [[2021]]},
        }
    }


class _CountingTransport:
    """Thread-safe httpx.Client stand-in that counts real (uncached) GETs.

    Maps each requested URL to a canned 200 JSON ``message`` payload. Every call
    is a genuine HTTP round-trip from the client's perspective; the SqliteCache
    in ``HttpClient._request`` is what eliminates the duplicates. So the call
    count == number of distinct cache misses, which is exactly what the warming
    optimization is meant to reduce on the per-entry path.
    """

    def __init__(self, routes: dict[str, dict]):
        # Crossref get_by_doi lowercases the DOI before building the /works URL,
        # so match needles case-insensitively.
        self._routes = {k.lower(): v for k, v in routes.items()}
        self.count = 0
        self._lock = threading.Lock()

    def request(self, method, url, **kwargs):
        # HttpClient._request calls client.request(..., json=json_body); accept
        # **kwargs so that keyword (and params/headers) is swallowed without
        # shadowing the module-level ``json`` used for serialization below.
        with self._lock:
            self.count += 1
        url = str(url).lower()
        for needle, payload in self._routes.items():
            if needle in url:
                return httpx.Response(
                    200,
                    content=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    request=httpx.Request(method, url),
                )
        return httpx.Response(
            404,
            content=b"{}",
            headers={"Content-Type": "application/json"},
            request=httpx.Request(method, url),
        )


def _make_http(routes: dict[str, dict], tmp_path) -> tuple[HttpClient, _CountingTransport]:
    """Real HttpClient with a real SqliteCache, but a counting transport stub.

    The cache is genuine so warming truly populates it and per-entry calls hit
    it; only the network layer is faked. Rate limits are set high so the limiter
    never sleeps for these tiny workloads.
    """
    cache = SqliteCache(str(tmp_path / "cache.db"))
    registry = RateLimiterRegistry(dict.fromkeys(RateLimiterRegistry.DEFAULT_LIMITS, 100_000))
    http = HttpClient(timeout=5.0, user_agent="test", rate_limiter=registry, cache=cache)
    transport = _CountingTransport(routes)
    http.client = transport  # type: ignore[assignment]
    return http, transport


def _ibrnet_entry() -> dict[str, str]:
    return {
        "ID": "ibrnet2021",
        "ENTRYTYPE": "inproceedings",
        "title": "IBRNet: Learning Multi-View Image-Based Rendering",
        "author": "Wang, Qianqian and Wang, Zhicheng and Genova, Kyle",
        "doi": "10.1109/CVPR46437.2021.00469",
        "year": "2021",
    }


def _imagebind_entry() -> dict[str, str]:
    return {
        "ID": "imagebind2023",
        "ENTRYTYPE": "inproceedings",
        "title": "ImageBind: One Embedding Space To Bind Them All",
        "author": "Girdhar, Rohit and El-Nouby, Alaaeldin",
        "doi": "10.1109/CVPR52729.2023.01457",
        "year": "2023",
    }


def _checker(http: HttpClient, logger, config: FactCheckerConfig | None = None) -> FactChecker:
    return FactChecker(
        CrossrefClient(http),
        DBLPClient(http),
        SemanticScholarClient(http),
        config or FactCheckerConfig(),
        logger,
    )


@pytest.fixture
def logger():
    return logging.getLogger("test_crossref_prefetch")


class TestPrefetchWarmsCache:
    """The pre-fetch issues each distinct DOI's /works fetch once; the per-entry
    get_by_doi calls then read from the warmed SqliteCache (zero extra GETs)."""

    def test_warm_then_per_entry_calls_are_cache_hits(self, logger, tmp_path):
        # Two entries, each with a wrong DOI (-> DOI_MISMATCH), distinct DOIs.
        routes = {
            "CVPR46437.2021.00469": _works_payload(
                "Delving into Localization Errors for Monocular 3D Object Detection",
                "10.1109/CVPR46437.2021.00469",
            ),
            "CVPR52729.2023.01457": _works_payload(
                "Segment Anything in High Quality", "10.1109/CVPR52729.2023.01457"
            ),
        }
        http, transport = _make_http(routes, tmp_path)
        checker = _checker(http, logger)

        # Warm the cache via the processor (no worker pool needed for the count).
        processor = FactCheckProcessor(checker, logger)
        entries = [_ibrnet_entry(), _imagebind_entry()]
        warmed = processor._batch_warm_crossref_records(entries)
        assert warmed == 2
        # Exactly one real GET per distinct DOI during warming.
        assert transport.count == 2

        # Now the per-entry consistency checks must add ZERO further GETs: every
        # get_by_doi is served from the warmed cache.
        before = transport.count
        for entry in entries:
            checker._check_doi_consistency(entry)
        assert transport.count == before

    def test_duplicate_dois_warmed_once(self, logger, tmp_path):
        """Identical DOIs across entries share one cache key -> one GET total."""
        routes = {
            "CVPR46437.2021.00469": _works_payload(
                "Delving into Localization Errors for Monocular 3D Object Detection",
                "10.1109/CVPR46437.2021.00469",
            )
        }
        http, transport = _make_http(routes, tmp_path)
        checker = _checker(http, logger)
        processor = FactCheckProcessor(checker, logger)

        warmed = processor._batch_warm_crossref_records([_ibrnet_entry(), _ibrnet_entry()])

        assert warmed == 1
        assert transport.count == 1

    def test_no_dois_no_fetch(self, logger, tmp_path):
        http, transport = _make_http({}, tmp_path)
        checker = _checker(http, logger)
        processor = FactCheckProcessor(checker, logger)
        entry = _ibrnet_entry()
        del entry["doi"]

        assert processor._batch_warm_crossref_records([entry]) == 0
        assert transport.count == 0

    def test_warm_noop_without_cache(self, logger):
        """No shared response cache -> warming is a documented no-op (the per-entry
        calls would re-fetch regardless, so warming buys nothing)."""
        http_no_cache = MagicMock()
        http_no_cache.cache = None
        checker = FactChecker(
            CrossrefClient(http_no_cache),
            DBLPClient(http_no_cache),
            SemanticScholarClient(http_no_cache),
            FactCheckerConfig(),
            logger,
        )
        processor = FactCheckProcessor(checker, logger)

        assert processor._batch_warm_crossref_records([_ibrnet_entry()]) == 0
        http_no_cache._request.assert_not_called()


class TestVerdictNeutral:
    """The verdicts (statuses) must be IDENTICAL with and without the warming;
    the optimization only changes WHEN the same records are fetched."""

    def _run(self, with_warm: bool, logger, tmp_path) -> list[FactCheckStatus]:
        routes = {
            # Wrong DOI -> DOI_MISMATCH for IBRNet.
            "CVPR46437.2021.00469": _works_payload(
                "Delving into Localization Errors for Monocular 3D Object Detection",
                "10.1109/CVPR46437.2021.00469",
            ),
            # Wrong DOI -> DOI_MISMATCH for ImageBind.
            "CVPR52729.2023.01457": _works_payload(
                "Segment Anything in High Quality", "10.1109/CVPR52729.2023.01457"
            ),
        }
        http, _ = _make_http(routes, tmp_path)
        checker = _checker(http, logger)
        processor = FactCheckProcessor(checker, logger)
        entries = [_ibrnet_entry(), _imagebind_entry()]
        if with_warm:
            processor._batch_warm_crossref_records(entries)
        # Drive the SAME per-entry path both ways.
        return [checker.check_entry(e).status for e in entries]

    def test_statuses_identical_with_and_without_warm(self, logger, tmp_path):
        cold = self._run(False, logger, tmp_path / "cold")
        warm = self._run(True, logger, tmp_path / "warm")

        assert cold == warm
        assert cold == [FactCheckStatus.DOI_MISMATCH, FactCheckStatus.DOI_MISMATCH]

    def test_process_entries_statuses_match_unwarmed_baseline(self, logger, tmp_path):
        """End-to-end via process_entries (which warms) vs a cold per-entry run:
        same statuses in the same order."""
        routes = {
            "CVPR46437.2021.00469": _works_payload(
                "Delving into Localization Errors for Monocular 3D Object Detection",
                "10.1109/CVPR46437.2021.00469",
            ),
            "CVPR52729.2023.01457": _works_payload(
                "Segment Anything in High Quality", "10.1109/CVPR52729.2023.01457"
            ),
        }
        entries = [_ibrnet_entry(), _imagebind_entry()]

        # Cold baseline: never warm, check each entry directly.
        http_cold, _ = _make_http(routes, tmp_path / "cold")
        cold_checker = _checker(http_cold, logger)
        cold = [cold_checker.check_entry(e).status for e in entries]

        # Warmed path through the real process_entries pipeline.
        http_warm, _ = _make_http(routes, tmp_path / "warm")
        warm_checker = _checker(http_warm, logger)
        warm_proc = FactCheckProcessor(warm_checker, logger)
        warm_results = warm_proc.process_entries(entries, max_workers=2)
        warm = [r.status for r in warm_results]

        assert warm == cold

    def test_correct_doi_verifies_identically(self, logger, tmp_path):
        """A CORRECT DOI (resolves to the cited paper) must VERIFY identically
        whether or not its record was pre-fetched -- warming never relaxes or
        tightens the comparison, only moves the fetch."""
        entry = _ibrnet_entry()
        authors = [
            {"given": "Qianqian", "family": "Wang"},
            {"given": "Zhicheng", "family": "Wang"},
            {"given": "Kyle", "family": "Genova"},
        ]
        # The DOI's /works record (and the title search) both return the cited
        # paper, so the entry VERIFIES (consistency passes, then full match).
        record = {
            "message": {
                "DOI": entry["doi"],
                "type": "proceedings-article",
                "title": [entry["title"]],
                "author": authors,
                "issued": {"date-parts": [[2021]]},
            }
        }
        routes = {"CVPR46437.2021.00469": record}

        def run(with_warm: bool, sub: str) -> FactCheckStatus:
            http, _ = _make_http(routes, tmp_path / sub)
            checker = _checker(http, logger)
            # Title search returns the same record so the entry can VERIFY.
            checker.crossref.search = MagicMock(return_value=[record["message"]])  # type: ignore[method-assign]
            if with_warm:
                FactCheckProcessor(checker, logger)._batch_warm_crossref_records([entry])
            return checker.check_entry(entry).status

        cold = run(False, "cold")
        warm = run(True, "warm")
        assert cold == warm
        assert cold == FactCheckStatus.VERIFIED

    def test_unindexed_doi_identical_verdict(self, logger, tmp_path):
        """A DOI Crossref does not index (the /works fetch 404s consistently) is
        'cannot determine' for the consistency check -> no flag, both warmed and
        unwarmed. Warming the 404 cannot change the verdict because the per-entry
        fetch would have hit the same 404."""

        def run(with_warm: bool, sub: str) -> FactCheckStatus:
            # No route for this DOI -> every /works GET 404s.
            http, _ = _make_http({}, tmp_path / sub)
            checker = _checker(http, logger)
            entries = [_ibrnet_entry()]
            if with_warm:
                FactCheckProcessor(checker, logger)._batch_warm_crossref_records(entries)
            return checker.check_entry(entries[0]).status

        cold = run(False, "cold")
        warm = run(True, "warm")
        assert cold == warm
        # No candidates from any source (all 404) and no DOI flag -> NOT_FOUND,
        # the FPR-safe abstention. Identical with/without warming.
        assert cold == FactCheckStatus.NOT_FOUND


class TestPrefetchFailureFallback:
    """A DOI whose pre-fetch fails (network error / non-200) is simply not cached;
    the per-entry get_by_doi then performs its own fetch -- correct result, no
    behavior change for that DOI."""

    def test_failed_warm_still_correct_per_entry(self, logger, tmp_path):
        """A DOI whose WARM round-trip transiently fails (raises, nothing cached)
        still gets its correct per-entry verdict once the endpoint is reachable.

        Models a real transient blip: ImageBind's get_by_doi raises during the
        up-front warm (so the cache is NOT poisoned), then succeeds on the
        per-entry call. IBRNet warms normally. Both must still flag DOI_MISMATCH.
        """
        http, _ = _make_http(
            {
                "CVPR46437.2021.00469": _works_payload(
                    "Delving into Localization Errors for Monocular 3D Object Detection",
                    "10.1109/CVPR46437.2021.00469",
                ),
                "CVPR52729.2023.01457": _works_payload(
                    "Segment Anything in High Quality", "10.1109/CVPR52729.2023.01457"
                ),
            },
            tmp_path,
        )
        checker = _checker(http, logger)
        processor = FactCheckProcessor(checker, logger)

        # ImageBind's DOI raises during warming only; restored before per-entry.
        real_get = checker.crossref.get_by_doi
        warm_phase = {"active": True}

        def flaky(doi: str):
            if warm_phase["active"] and "CVPR52729" in doi:
                raise RuntimeError("transient warm failure")
            return real_get(doi)

        checker.crossref.get_by_doi = MagicMock(side_effect=flaky)  # type: ignore[method-assign]

        warmed = processor._batch_warm_crossref_records([_ibrnet_entry(), _imagebind_entry()])
        assert warmed == 2  # both attempted; ImageBind's failure swallowed

        # Per-entry phase: the failed-warm DOI now reaches the endpoint and is
        # fetched correctly -> same DOI_MISMATCH verdict as the warmed one.
        warm_phase["active"] = False
        assert checker.check_entry(_ibrnet_entry()).status == FactCheckStatus.DOI_MISMATCH
        assert checker.check_entry(_imagebind_entry()).status == FactCheckStatus.DOI_MISMATCH

    def test_warm_swallows_get_by_doi_exception(self, logger, tmp_path):
        """An exception thrown for one DOI during warming never aborts the batch
        and the entry still gets its correct per-entry verdict."""
        http, _ = _make_http(
            {
                "CVPR46437.2021.00469": _works_payload(
                    "Delving into Localization Errors for Monocular 3D Object Detection",
                    "10.1109/CVPR46437.2021.00469",
                )
            },
            tmp_path,
        )
        checker = _checker(http, logger)
        processor = FactCheckProcessor(checker, logger)

        # Make get_by_doi blow up for ImageBind's DOI only, during warming.
        real_get = checker.crossref.get_by_doi

        def flaky(doi: str):
            if "CVPR52729" in doi:
                raise RuntimeError("transient warm failure")
            return real_get(doi)

        checker.crossref.get_by_doi = MagicMock(side_effect=flaky)  # type: ignore[method-assign]

        # Warming must not raise even though one DOI errors.
        warmed = processor._batch_warm_crossref_records([_ibrnet_entry(), _imagebind_entry()])
        assert warmed == 2

        # IBRNet warmed normally -> mismatch verdict from per-entry path.
        checker.crossref.get_by_doi.side_effect = lambda doi: real_get(doi)
        assert checker.check_entry(_ibrnet_entry()).status == FactCheckStatus.DOI_MISMATCH
