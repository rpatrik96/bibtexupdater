"""Tests for the relaxed-author retrieval fallback (FIX X4).

For DOI-less, title-and-author-strict cascade queries that returned zero
usable candidates, the cascade today exits ``not_found``. For HALLUCINATED
entries that hides the hallucination (the tool can't tell hallucinated
apart from new + niche).

The fix retries Crossref + OpenAlex with title-only (no author param) when
the standard cascade returns zero usable candidates. The new candidates
flow through the same ``_score_candidate`` + ``_has_full_confirmation`` +
``_determine_status`` gates as the regular cascade. The realistic
transition is ``not_found -> AUTHOR_MISMATCH`` (the fallback finds a
wrong-paper candidate whose title is close enough but whose authors
disagree); never ``not_found -> VERIFIED`` (VERIFIED requires strong
title AND author confirmation regardless of retrieval path).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from bibtex_updater.fact_checker import (
    CrossrefClient,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    FactCheckStatus,
    SemanticScholarClient,
)


def _crossref_message(
    title: str,
    doi: str,
    *,
    authors: list[dict] | None = None,
    venue: str | None = None,
    year: int | None = None,
) -> dict:
    msg: dict = {
        "DOI": doi,
        "type": "proceedings-article",
        "title": [title],
        "author": authors or [{"given": "A", "family": "Author"}],
    }
    if venue:
        msg["container-title"] = [venue]
    if year:
        msg["issued"] = {"date-parts": [[year]]}
    return msg


@pytest.fixture
def logger():
    return logging.getLogger("test_relaxed_author_fallback")


@pytest.fixture
def empty_http():
    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return mock


@pytest.fixture
def dead_sources(empty_http):
    return (
        CrossrefClient(empty_http),
        DBLPClient(empty_http),
        SemanticScholarClient(empty_http),
    )


class TestRelaxedAuthorRetrievalFallback:
    """The fallback runs ONCE per entry when the standard cascade produces
    zero usable candidates. It tags candidates with the ``-fallback`` source
    suffix so the relaxed-retrieval provenance is observable.
    """

    def test_cascade_verifies_fallback_not_taken(self, dead_sources, logger):
        """Negative gate: when the standard cascade already finds the paper,
        the fallback path must NOT run -- the cache hit prevents the second
        query."""
        crossref, dblp, s2 = dead_sources
        entry = {
            "ID": "verified",
            "ENTRYTYPE": "inproceedings",
            "title": "A Real Paper",
            "author": "Smith, John",
            "booktitle": "ICML",
            "year": "2021",
        }
        authors = [{"given": "John", "family": "Smith"}]
        crossref.search = MagicMock(
            return_value=[
                _crossref_message(entry["title"], "10.0/x", authors=authors, venue="ICML", year=2021)
            ]
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)

        assert result.status == FactCheckStatus.VERIFIED
        # No fallback source should appear when the cascade already verified.
        assert "crossref-fallback" not in result.api_sources_queried
        assert "openalex-fallback" not in result.api_sources_queried

    def test_hallucinated_entry_routes_to_author_mismatch(self, dead_sources, logger):
        """Positive: an entry whose strict cascade returns zero candidates
        triggers the fallback. The title-only retry surfaces a wrong-paper
        candidate (same title, different authors). The scoring path then
        routes it to AUTHOR_MISMATCH -- never to VERIFIED, because the
        author gate is independent of the retrieval path."""
        crossref, dblp, s2 = dead_sources
        entry = {
            "ID": "hallucinated",
            "ENTRYTYPE": "inproceedings",
            "title": "Some Specific Real Paper Title",
            "author": "Hallucinated, A. and Imaginary, B.",
            "booktitle": "ICML",
            "year": "2026",
        }
        # Track which search calls pass author= so we can verify the fallback
        # actually dropped the author parameter on the retry.
        call_log: list[dict] = []

        def _search(query, rows=10, title=None, author=None):
            call_log.append({"title": title, "author": author, "rows": rows})
            # First call (strict, with author) -> zero hits.
            if author:
                return []
            # Second call (relaxed, title-only) -> wrong-paper candidate.
            return [
                _crossref_message(
                    "Some Specific Real Paper Title",
                    "10.0/different",
                    authors=[{"given": "Real", "family": "Author"}],
                    venue="ICML",
                    year=2021,
                )
            ]

        crossref.search = MagicMock(side_effect=_search)
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)

        # The fallback fired and surfaced a wrong-paper candidate.
        assert "crossref-fallback" in result.api_sources_queried
        # The wrong-paper candidate has the right title but wrong authors,
        # so the existing gates route it to AUTHOR_MISMATCH (NOT VERIFIED).
        assert result.status != FactCheckStatus.VERIFIED
        assert result.status in (
            FactCheckStatus.AUTHOR_MISMATCH,
            FactCheckStatus.PARTIAL_MATCH,
            FactCheckStatus.HALLUCINATED,
        )
        # And critically: the second call did NOT pass an author param.
        assert any(c["author"] is None for c in call_log), call_log

    def test_fallback_does_not_verify_a_wrong_paper(self, dead_sources, logger):
        """The key FPR guard: the fallback never converts a hallucinated entry
        into a VERIFIED verdict. The scoring + status gates demand strong
        title AND author confirmation; the relaxed retrieval only widens
        the candidate pool."""
        crossref, dblp, s2 = dead_sources
        entry = {
            "ID": "would_be_leak",
            "ENTRYTYPE": "inproceedings",
            "title": "An Exact Title That Exists",
            "author": "FakeAuthor, X. and OtherFake, Y.",
            "booktitle": "ICML",
            "year": "2021",
        }
        call_count = {"n": 0}

        def _search(query, rows=10, title=None, author=None):
            call_count["n"] += 1
            if author:
                return []  # strict pass returns nothing
            # Relaxed pass: same title, completely different authors.
            return [
                _crossref_message(
                    "An Exact Title That Exists",
                    "10.0/real",
                    authors=[
                        {"given": "Real", "family": "Researcher"},
                        {"given": "Another", "family": "RealCoauthor"},
                    ],
                    venue="ICML",
                    year=2021,
                )
            ]

        crossref.search = MagicMock(side_effect=_search)
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)

        assert result.status != FactCheckStatus.VERIFIED
        assert call_count["n"] >= 2  # cascade + fallback ran

    def test_fallback_skipped_when_cascade_has_usable_candidate(self, dead_sources, logger):
        """When the cascade produces a candidate ABOVE ``abstention_below``,
        the fallback never runs -- the standard scoring path takes over."""
        crossref, dblp, s2 = dead_sources
        entry = {
            "ID": "found_via_cascade",
            "ENTRYTYPE": "inproceedings",
            "title": "Foundable Paper",
            "author": "Smith, John",
            "year": "2021",
        }
        authors = [{"given": "John", "family": "Smith"}]
        # The strict (author-included) search already returns a strong hit.
        crossref.search = MagicMock(
            return_value=[
                _crossref_message(entry["title"], "10.0/x", authors=authors, venue="ICML", year=2021)
            ]
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)

        assert "crossref-fallback" not in result.api_sources_queried

    def test_fallback_runs_only_once_per_entry(self, dead_sources, logger):
        """The fallback is gated to a single retry per entry. The
        ``crossref-fallback`` source name appears at most once in the queried
        list."""
        crossref, dblp, s2 = dead_sources
        entry = {
            "ID": "single_retry",
            "ENTRYTYPE": "inproceedings",
            "title": "Title With No Hits",
            "author": "Nobody, N.",
            "year": "2026",
        }
        # All searches return nothing. The fallback still runs (once) and
        # returns nothing too.
        crossref.search = MagicMock(return_value=[])
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)

        assert result.api_sources_queried.count("crossref-fallback") == 1
        # Empty fallback -> still NOT_FOUND.
        assert result.status == FactCheckStatus.NOT_FOUND
