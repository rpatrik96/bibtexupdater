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

# The real ONEBench paper (arXiv:2412.07689).
ONEBENCH_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2412.07689v1</id>
    <published>2024-12-10T18:00:00Z</published>
    <title>ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities</title>
    <summary>We propose ONEBench...</summary>
    <author><name>Adhiraj Ghosh</name></author>
    <author><name>Sebastian Dziadzio</name></author>
    <author><name>Ameya Prabhu</name></author>
    <author><name>Vishaal Udandarao</name></author>
    <author><name>Samuel Albanie</name></author>
    <author><name>Matthias Bethge</name></author>
    <arxiv:primary_category term="cs.LG"/>
  </entry>
</feed>
"""

# The *unrelated* paper that the wrong arXiv:2412.06745 actually points to.
ROBOTRON_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2412.06745v1</id>
    <published>2024-12-09T18:00:00Z</published>
    <title>RoboTron-Drive: All-in-One Large Multimodal Model for Autonomous Driving</title>
    <summary>We present RoboTron-Drive...</summary>
    <author><name>Zhijian Huang</name></author>
    <author><name>Chengjian Feng</name></author>
    <author><name>Feng Yan</name></author>
    <arxiv:primary_category term="cs.CV"/>
  </entry>
</feed>
"""


def _onebench_entry(arxiv_url: str) -> dict[str, str]:
    """ONEBench bib entry parameterized by the arXiv URL it cites."""
    return {
        "ID": "onebench2024",
        "ENTRYTYPE": "inproceedings",
        "title": "ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities",
        "author": (
            "Ghosh, Adhiraj and Dziadzio, Sebastian and Prabhu, Ameya and "
            "Udandarao, Vishaal and Albanie, Samuel and Bethge, Matthias"
        ),
        "url": arxiv_url,
        "year": "2024",
    }


class TestArxivIdConsistency:
    """Regression: an entry whose cited arXiv ID resolves to a *different* paper
    must be flagged ARXIV_ID_MISMATCH, not silently VERIFIED against the real
    paper found by title/author search. This is the onebench2024 failure where
    the correct arXiv:2412.07689 was replaced with the unrelated 2412.06745.
    """

    def test_wrong_arxiv_id_flagged_mismatch(self, dead_sources, logger):
        crossref, dblp, s2 = dead_sources
        arxiv = _StubArxiv({"2412.06745": ROBOTRON_ATOM})
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger, arxiv=arxiv)

        result = checker.check_entry(_onebench_entry("https://arxiv.org/abs/2412.06745"))

        assert result.status == FactCheckStatus.ARXIV_ID_MISMATCH
        assert "2412.06745" in arxiv.requested
        assert result.best_match is not None
        assert "RoboTron" in result.best_match.title
        assert result.errors and "2412.06745" in result.errors[0]

    def test_correct_arxiv_id_not_flagged(self, dead_sources, logger):
        """The correct ID resolves to ONEBench, so the entry verifies instead."""
        crossref, dblp, s2 = dead_sources
        arxiv = _StubArxiv({"2412.07689": ONEBENCH_ATOM})
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger, arxiv=arxiv)

        result = checker.check_entry(_onebench_entry("https://arxiv.org/abs/2412.07689"))

        assert result.status != FactCheckStatus.ARXIV_ID_MISMATCH
        assert result.status == FactCheckStatus.VERIFIED

    def test_mismatch_check_disabled_by_config(self, dead_sources, logger):
        """With the guard disabled, the wrong ID is not flagged (opt-out works)."""
        crossref, dblp, s2 = dead_sources
        arxiv = _StubArxiv({"2412.06745": ROBOTRON_ATOM})
        config = FactCheckerConfig(check_arxiv_consistency=False)
        checker = FactChecker(crossref, dblp, s2, config, logger, arxiv=arxiv)

        result = checker.check_entry(_onebench_entry("https://arxiv.org/abs/2412.06745"))

        assert result.status != FactCheckStatus.ARXIV_ID_MISMATCH

    def test_no_arxiv_client_skips_consistency_check(self, dead_sources, logger):
        """Without an arXiv client the guard is a no-op (no false mismatch)."""
        crossref, dblp, s2 = dead_sources
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(_onebench_entry("https://arxiv.org/abs/2412.06745"))

        assert result.status != FactCheckStatus.ARXIV_ID_MISMATCH


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
