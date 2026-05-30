"""Tests for ID-anchored venue/year mismatch on DOI-resolved records (FIX X3).

For the 143-entry "HALLUCINATED + has DOI + unconfirmed" cluster, the
dominant pattern is: DOI resolves to the real paper (Crossref agrees on
title + authors), but the venue or year claimed by the entry is wrong
(e.g. AAAI cited as AISTATS, NeurIPS cited as ICLR, year shifted from
2021 to 2026). The cascade abstained because no existing helper flagged
venue/year on the DOI-resolved record.

The fix adds ``_id_anchored_field_mismatch`` (mirroring
``_id_anchored_author_mismatch``). Called from ``_check_doi_consistency``
after a title-confirming DOI fetch, it flags:
  * Hard venue MISMATCH -> VENUE_MISMATCH (preprint records abstain)
  * Hard year MISMATCH beyond tolerance on a non-preprint record
    -> YEAR_MISMATCH

FPR-safe gating: NON_COMPARABLE venues (preprint/series) never anchor;
blank entry venue/year abstain; preprint-twin DOIs cannot anchor years.
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
    """Crossref ``works`` message tailored for ``crossref_message_to_record``."""
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
    return logging.getLogger("test_id_anchored_field_mismatch")


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


def _aaai_entry() -> dict[str, str]:
    """An entry that cites a real AAAI paper but at the wrong conference."""
    return {
        "ID": "wrong_venue_aaai",
        "ENTRYTYPE": "inproceedings",
        "title": "A Real AAAI Paper",
        "author": "Smith, John and Doe, Jane",
        "booktitle": "AISTATS",  # Wrong: the real venue is AAAI.
        "doi": "10.1609/aaai.v35i1.16100",
        "year": "2021",
    }


def _year_shifted_entry() -> dict[str, str]:
    """An entry that cites a real 2021 paper at year 2026."""
    return {
        "ID": "year_shift",
        "ENTRYTYPE": "inproceedings",
        "title": "A Real Paper",
        "author": "Smith, John",
        "booktitle": "CVPR",
        "doi": "10.1109/CVPR46437.2021.00123",
        "year": "2026",  # Wrong: the real year is 2021.
    }


class TestVenueMismatchOnDoiResolvedRecord:
    """DOI title confirms + venue hard-MISMATCHes -> VENUE_MISMATCH."""

    def test_aaai_venue_cited_as_aistats_flags(self, dead_sources, logger):
        """Positive: AAAI paper cited as AISTATS, DOI returns the same paper
        with venue=AAAI. The fix should fire VENUE_MISMATCH."""
        crossref, dblp, s2 = dead_sources
        authors = [
            {"given": "John", "family": "Smith"},
            {"given": "Jane", "family": "Doe"},
        ]
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(
                "A Real AAAI Paper",
                "10.1609/aaai.v35i1.16100",
                authors=authors,
                venue="AAAI",
                year=2021,
            )
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(_aaai_entry())

        assert result.status == FactCheckStatus.VENUE_MISMATCH
        assert result.best_match is not None
        assert "AAAI" in (result.best_match.journal or "")

    def test_preprint_venue_on_record_abstains(self, dead_sources, logger):
        """Negative: DOI resolves to a record whose venue is a preprint/series
        (NON_COMPARABLE) -> the helper abstains (preserves FIX 3 / FIX 5)."""
        crossref, dblp, s2 = dead_sources
        authors = [
            {"given": "John", "family": "Smith"},
            {"given": "Jane", "family": "Doe"},
        ]
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(
                "A Real AAAI Paper",
                "10.1609/aaai.v35i1.16100",
                authors=authors,
                venue="arXiv preprint arXiv:2010.12345",
                year=2021,
            )
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(_aaai_entry())

        # Preprint record cannot anchor a published venue claim.
        assert result.status != FactCheckStatus.VENUE_MISMATCH

    def test_blank_entry_venue_abstains(self, dead_sources, logger):
        """Negative: entry makes no venue claim -> nothing to refute."""
        crossref, dblp, s2 = dead_sources
        entry = _aaai_entry()
        del entry["booktitle"]
        authors = [
            {"given": "John", "family": "Smith"},
            {"given": "Jane", "family": "Doe"},
        ]
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(
                "A Real AAAI Paper",
                "10.1609/aaai.v35i1.16100",
                authors=authors,
                venue="AAAI",
                year=2021,
            )
        )
        crossref.search = MagicMock(
            return_value=[
                _crossref_message(
                    "A Real AAAI Paper",
                    "10.1609/aaai.v35i1.16100",
                    authors=authors,
                    venue="AAAI",
                    year=2021,
                )
            ]
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)

        assert result.status != FactCheckStatus.VENUE_MISMATCH


class TestYearMismatchOnDoiResolvedRecord:
    """DOI title confirms + year hard-MISMATCHes beyond tolerance on a
    non-preprint record -> YEAR_MISMATCH."""

    def test_year_shifted_beyond_tolerance_flags(self, dead_sources, logger):
        """Positive: entry year 2026, DOI record year 2021, same paper,
        non-preprint venue (CVPR) -> YEAR_MISMATCH."""
        crossref, dblp, s2 = dead_sources
        authors = [{"given": "John", "family": "Smith"}]
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(
                "A Real Paper",
                "10.1109/CVPR46437.2021.00123",
                authors=authors,
                venue="CVPR",
                year=2021,
            )
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(_year_shifted_entry())

        assert result.status == FactCheckStatus.YEAR_MISMATCH

    def test_year_within_tolerance_does_not_flag(self, dead_sources, logger):
        """Negative: 1-year drift is within default tolerance -> no flag."""
        crossref, dblp, s2 = dead_sources
        entry = _year_shifted_entry()
        entry["year"] = "2022"  # only 1 year off from record's 2021
        authors = [{"given": "John", "family": "Smith"}]
        matching = _crossref_message(
            "A Real Paper",
            "10.1109/CVPR46437.2021.00123",
            authors=authors,
            venue="CVPR",
            year=2021,
        )
        crossref.get_by_doi = MagicMock(return_value=matching)
        crossref.search = MagicMock(return_value=[matching])
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)

        assert result.status != FactCheckStatus.YEAR_MISMATCH

    def test_preprint_record_does_not_anchor_year(self, dead_sources, logger):
        """Negative: DOI record is a preprint (``_doi_is_preprint`` True) ->
        cannot anchor a published year; abstain even on a large gap."""
        crossref, dblp, s2 = dead_sources
        authors = [{"given": "John", "family": "Smith"}]
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(
                "A Real Paper",
                "10.48550/arXiv.2010.12345",  # arXiv DataCite DOI -> preprint
                authors=authors,
                venue="arXiv",
                year=2020,
            )
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)
        entry = _year_shifted_entry()
        entry["doi"] = "10.48550/arXiv.2010.12345"

        result = checker.check_entry(entry)

        # Preprint record cannot anchor a published year mismatch.
        assert result.status != FactCheckStatus.YEAR_MISMATCH

    def test_blank_entry_year_abstains(self, dead_sources, logger):
        """Negative: entry has no year claim -> nothing to refute."""
        crossref, dblp, s2 = dead_sources
        entry = _year_shifted_entry()
        del entry["year"]
        # Pre-API year validation skips with no year; proceed to consistency check.
        authors = [{"given": "John", "family": "Smith"}]
        matching = _crossref_message(
            "A Real Paper",
            "10.1109/CVPR46437.2021.00123",
            authors=authors,
            venue="CVPR",
            year=2021,
        )
        crossref.get_by_doi = MagicMock(return_value=matching)
        crossref.search = MagicMock(return_value=[matching])
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)

        assert result.status != FactCheckStatus.YEAR_MISMATCH


class TestPrecedenceAndSuppressionGates:
    """The new helper sits behind the title-confirm gate and behind the
    author-mismatch helper; both must keep their existing behaviour."""

    def test_doi_title_does_not_confirm_suppresses(self, dead_sources, logger):
        """Negative: DOI title differs significantly -> the existing
        DOI_MISMATCH path handles it; the new helper never runs (it's
        only called from the title-confirm branch)."""
        crossref, dblp, s2 = dead_sources
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(
                "A Completely Different Paper",  # title does NOT confirm
                "10.1609/aaai.v35i1.16100",
                venue="AAAI",
                year=2021,
            )
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(_aaai_entry())

        # The pre-existing DOI_MISMATCH path handles this case.
        assert result.status == FactCheckStatus.DOI_MISMATCH

    def test_author_mismatch_takes_priority_over_venue(self, dead_sources, logger):
        """Positive: when BOTH authors and venue mismatch on the DOI-resolved
        record, AUTHOR_MISMATCH takes precedence (mirrors the existing
        ``_determine_status`` priority for ID-anchored findings)."""
        crossref, dblp, s2 = dead_sources
        # Title confirms, authors DIFFER, venue ALSO differs.
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(
                "A Real AAAI Paper",
                "10.1609/aaai.v35i1.16100",
                authors=[{"given": "Different", "family": "Author"}],
                venue="AAAI",
                year=2021,
            )
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(_aaai_entry())

        # Author mismatch wins (stronger hallucination signal).
        assert result.status == FactCheckStatus.AUTHOR_MISMATCH

    def test_authors_and_venue_match_verifies(self, dead_sources, logger):
        """Negative: when everything (title + authors + venue + year)
        confirms on the DOI-resolved record, no flag fires; the cascade
        proceeds and the entry verifies."""
        crossref, dblp, s2 = dead_sources
        entry = _aaai_entry()
        entry["booktitle"] = "AAAI"  # correct venue
        authors = [
            {"given": "John", "family": "Smith"},
            {"given": "Jane", "family": "Doe"},
        ]
        matching = _crossref_message(
            entry["title"],
            entry["doi"],
            authors=authors,
            venue="AAAI",
            year=2021,
        )
        crossref.get_by_doi = MagicMock(return_value=matching)
        crossref.search = MagicMock(return_value=[matching])
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)

        assert result.status == FactCheckStatus.VERIFIED
