"""Tests for arXiv-ID based verification of academic preprints.

Regression coverage for the gap where ``bibtex-check`` verified entries purely
by title/author text search. Brand-new arXiv-only preprints are not yet indexed
by Crossref/DBLP/Semantic Scholar, so that search returned garbage and the
checker emitted a false ``HALLUCINATED`` (or ``NOT_FOUND``) verdict. When an
entry carries an arXiv identifier (``eprint``/``archivePrefix`` or an
``arxiv.org/abs`` URL) the checker now fetches the authoritative arXiv record by
ID and uses it as a verification candidate.
"""

from __future__ import annotations

import logging

import pytest

from bibtex_updater.fact_checker import (
    CrossrefClient,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    FactCheckStatus,
    SemanticScholarClient,
)
from bibtex_updater.utils import arxiv_atom_to_record

# Trimmed but structurally faithful arXiv Atom response for id_list=2602.01031.
HALLUHARD_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title type="html">ArXiv Query: search_query=&amp;id_list=2602.01031</title>
  <id>http://arxiv.org/api/abc</id>
  <entry>
    <id>http://arxiv.org/abs/2602.01031v1</id>
    <updated>2026-02-01T18:00:00Z</updated>
    <published>2026-02-01T18:00:00Z</published>
    <title>HalluHard: A Hard Multi-Turn Hallucination Benchmark</title>
    <summary>We introduce a benchmark...</summary>
    <author><name>Dongyang Fan</name></author>
    <author><name>Sebastien Delsad</name></author>
    <author><name>Nicolas Flammarion</name></author>
    <author><name>Maksym Andriushchenko</name></author>
    <arxiv:primary_category term="cs.CL"/>
  </entry>
</feed>
"""

# arXiv returns a 200 feed with a single sentinel "Error" entry for bad IDs.
ARXIV_ERROR_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <id>http://arxiv.org/api/errors</id>
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_2602.99999</id>
    <title>Error</title>
    <summary>incorrect id format for 2602.99999</summary>
  </entry>
</feed>
"""


class _StubArxiv:
    """ArxivClient stand-in returning canned Atom XML, no network."""

    def __init__(self, atom_by_id: dict[str, str]):
        self._atom_by_id = atom_by_id
        self.requested: list[str] = []

    def fetch_atom(self, arxiv_id: str) -> str | None:
        self.requested.append(arxiv_id)
        return self._atom_by_id.get(arxiv_id)


@pytest.fixture
def logger():
    return logging.getLogger("test_arxiv_id_lookup")


@pytest.fixture
def empty_http():
    """HTTP client whose every request looks like a 404/empty result."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return mock


@pytest.fixture
def dead_sources(empty_http):
    """Crossref/DBLP/S2 clients that return nothing (mimic un-indexed preprint)."""
    return (
        CrossrefClient(empty_http),
        DBLPClient(empty_http),
        SemanticScholarClient(empty_http),
    )


def _halluhard_entry() -> dict[str, str]:
    # Authors carry the wrong *given* names exactly as they appeared in the
    # paper; surnames are correct. The checker compares surnames, so this is a
    # realistic "valid preprint" entry.
    return {
        "ID": "halluhard2026",
        "ENTRYTYPE": "misc",
        "title": "HalluHard: A Hard Multi-Turn Hallucination Benchmark",
        "author": "Fan, Yiwei and Delsad, Maxime and Flammarion, Nicolas and Andriushchenko, Maksym",
        "eprint": "2602.01031",
        "archiveprefix": "arXiv",
        "url": "https://arxiv.org/abs/2602.01031",
        "year": "2026",
    }


class TestArxivAtomParser:
    def test_parses_title_authors_year(self):
        rec = arxiv_atom_to_record(HALLUHARD_ATOM)
        assert rec is not None
        assert rec.title == "HalluHard: A Hard Multi-Turn Hallucination Benchmark"
        assert rec.year == 2026
        families = [a["family"] for a in rec.authors]
        assert families == ["Fan", "Delsad", "Flammarion", "Andriushchenko"]
        assert rec.authors[0] == {"given": "Dongyang", "family": "Fan"}

    def test_error_feed_returns_none(self):
        assert arxiv_atom_to_record(ARXIV_ERROR_ATOM) is None

    def test_garbage_returns_none(self):
        assert arxiv_atom_to_record("not xml at all") is None


class TestArxivIdVerification:
    def test_unindexed_preprint_verified_via_arxiv(self, dead_sources, logger):
        """The core regression: a valid arXiv-only preprint must not be flagged
        HALLUCINATED/NOT_FOUND just because text search misses it."""
        crossref, dblp, s2 = dead_sources
        arxiv = _StubArxiv({"2602.01031": HALLUHARD_ATOM})
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger, arxiv=arxiv)

        result = checker.check_entry(_halluhard_entry())

        assert result.status == FactCheckStatus.VERIFIED
        assert result.best_match is not None
        assert result.best_match.title == "HalluHard: A Hard Multi-Turn Hallucination Benchmark"
        assert "arxiv" in result.api_sources_queried
        assert "2602.01031" in arxiv.requested

    def test_without_arxiv_client_still_not_found(self, dead_sources, logger):
        """Guard: the flip is caused by the arXiv lookup, not something else."""
        crossref, dblp, s2 = dead_sources
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(_halluhard_entry())

        assert result.status in (FactCheckStatus.NOT_FOUND, FactCheckStatus.API_ERROR)

    def test_arxiv_id_extracted_from_url_only(self, dead_sources, logger):
        """arXiv ID is recovered from an arxiv.org/abs URL when eprint is absent."""
        crossref, dblp, s2 = dead_sources
        arxiv = _StubArxiv({"2602.01031": HALLUHARD_ATOM})
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger, arxiv=arxiv)

        entry = _halluhard_entry()
        del entry["eprint"]
        del entry["archiveprefix"]

        result = checker.check_entry(entry)
        assert result.status == FactCheckStatus.VERIFIED
        assert "2602.01031" in arxiv.requested

    def test_non_arxiv_entry_skips_lookup(self, dead_sources, logger):
        """Entries with no arXiv identifier never hit the arXiv client."""
        crossref, dblp, s2 = dead_sources
        arxiv = _StubArxiv({})
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger, arxiv=arxiv)

        entry = {
            "ID": "smith2020",
            "ENTRYTYPE": "article",
            "title": "Some Journal Paper Without A Preprint",
            "author": "Smith, John",
            "journal": "Journal of Things",
            "year": "2020",
        }
        checker.check_entry(entry)
        assert arxiv.requested == []
