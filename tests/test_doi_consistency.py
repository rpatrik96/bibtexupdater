"""Tests for DOI-target consistency checking in ``bibtex-check``.

Regression coverage for the gap where the tool only verified that an entry's
DOI *resolves* (doi.org HEAD; 404/410 -> DOI_NOT_FOUND) but never that the DOI
points to the *cited* paper. A copy-paste DOI that resolves to a completely
different work (e.g. "IBRNet" carrying DOI 10.1109/CVPR46437.2021.00469, which
belongs to "Delving into Localization Errors for Monocular 3D Object Detection")
otherwise survived because title/author search VERIFIES the entry against its
real record. ``_check_doi_consistency`` fetches the DOI's Crossref record and
flags a clear title mismatch as ``DOI_MISMATCH``.

All tests are hermetic: the Crossref get-by-DOI is mocked to return a controlled
title (or a failure), no network access.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from bibtex_updater.fact_checker import (
    ABSTAINED_STATUS_VALUES,
    CrossrefClient,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    FactCheckProcessor,
    FactCheckStatus,
    SemanticScholarClient,
)


def _crossref_message(title: str, doi: str, authors: list[dict] | None = None) -> dict:
    """Minimal Crossref ``works`` message for ``crossref_message_to_record``."""
    return {
        "DOI": doi,
        "type": "proceedings-article",
        "title": [title],
        "author": authors or [{"given": "A", "family": "Author"}],
    }


def _ibrnet_entry() -> dict[str, str]:
    """IBRNet entry whose DOI actually belongs to a 3D-detection paper."""
    return {
        "ID": "ibrnet2021",
        "ENTRYTYPE": "inproceedings",
        "title": "IBRNet: Learning Multi-View Image-Based Rendering",
        "author": "Wang, Qianqian and Wang, Zhicheng and Genova, Kyle",
        "doi": "10.1109/CVPR46437.2021.00469",
        "year": "2021",
    }


def _imagebind_entry() -> dict[str, str]:
    """ImageBind entry carrying a DOI for a different paper."""
    return {
        "ID": "imagebind2023",
        "ENTRYTYPE": "inproceedings",
        "title": "ImageBind: One Embedding Space To Bind Them All",
        "author": "Girdhar, Rohit and El-Nouby, Alaaeldin",
        "doi": "10.1109/CVPR52729.2023.01457",
        "year": "2023",
    }


class TestDoiConsistency:
    """An entry whose cited DOI resolves to a *different* paper must be flagged
    DOI_MISMATCH (positive evidence), not silently VERIFIED against the real
    paper found by title/author search.
    """

    def test_ibrnet_wrong_doi_flagged_mismatch(self, dead_sources, logger):
        crossref, dblp, s2 = dead_sources
        # DOI resolves to the 3D-detection paper, not IBRNet.
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(
                "Delving into Localization Errors for Monocular 3D Object Detection",
                "10.1109/CVPR46437.2021.00469",
            )
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(_ibrnet_entry())

        assert result.status == FactCheckStatus.DOI_MISMATCH
        crossref.get_by_doi.assert_called_once_with("10.1109/CVPR46437.2021.00469")
        assert result.best_match is not None
        assert "Localization Errors" in result.best_match.title
        assert result.errors and "Localization Errors" in result.errors[0]

    def test_doi_mismatch_in_problematic_bucket(self, dead_sources, logger):
        """DOI_MISMATCH is positive evidence -> problematic, not abstained."""
        crossref, dblp, s2 = dead_sources
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(
                "Delving into Localization Errors for Monocular 3D Object Detection",
                "10.1109/CVPR46437.2021.00469",
            )
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)
        processor = FactCheckProcessor(checker, logger)

        result = checker.check_entry(_ibrnet_entry())
        summary = processor.generate_summary([result])

        # Positive evidence, not abstention.
        assert FactCheckStatus.DOI_MISMATCH.value not in ABSTAINED_STATUS_VALUES
        assert summary["problematic_count"] == 1
        assert summary["abstained_count"] == 0
        assert summary["verified_count"] == 0

    def test_imagebind_wrong_doi_flagged_mismatch(self, dead_sources, logger):
        crossref, dblp, s2 = dead_sources
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(
                "Segment Anything in High Quality",
                "10.1109/CVPR52729.2023.01457",
            )
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(_imagebind_entry())

        assert result.status == FactCheckStatus.DOI_MISMATCH
        assert result.best_match is not None
        assert "Segment Anything" in result.best_match.title

    def test_correct_doi_not_flagged(self, dead_sources, logger):
        """A correct DOI resolves to the cited title -> no flag, entry verifies."""
        crossref, dblp, s2 = dead_sources
        entry = _ibrnet_entry()
        authors = [
            {"given": "Qianqian", "family": "Wang"},
            {"given": "Zhicheng", "family": "Wang"},
            {"given": "Kyle", "family": "Genova"},
        ]
        matching = _crossref_message(entry["title"], entry["doi"], authors=authors)
        matching["issued"] = {"date-parts": [[2021]]}
        # DOI resolves to the same paper (consistency check passes) and the
        # title search also returns it so the entry VERIFIES.
        crossref.get_by_doi = MagicMock(return_value=matching)
        crossref.search = MagicMock(return_value=[matching])
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)

        assert result.status != FactCheckStatus.DOI_MISMATCH
        assert result.status == FactCheckStatus.VERIFIED

    def test_no_doi_entry_returns_none(self, dead_sources, logger):
        """An entry without a DOI is never flagged for DOI mismatch."""
        crossref, dblp, s2 = dead_sources
        crossref.get_by_doi = MagicMock(return_value=_crossref_message("X", "10.0/x"))
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)
        entry = _ibrnet_entry()
        del entry["doi"]

        assert checker._check_doi_consistency(entry) is None
        crossref.get_by_doi.assert_not_called()

    def test_fetch_fails_returns_none(self, dead_sources, logger):
        """A failed/non-200 Crossref fetch (e.g. IEEE bot-block, un-indexed DOI)
        is 'cannot determine' -> no flag (FPR-safe)."""
        crossref, dblp, s2 = dead_sources
        crossref.get_by_doi = MagicMock(return_value=None)
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker._check_doi_consistency(_ibrnet_entry())

        assert result is None

    def test_consistency_check_disabled_by_config(self, dead_sources, logger):
        """With the guard disabled the wrong DOI is not flagged (opt-out works)."""
        crossref, dblp, s2 = dead_sources
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message("A Completely Different Paper", "10.0/x")
        )
        config = FactCheckerConfig(check_doi_consistency=False)
        checker = FactChecker(crossref, dblp, s2, config, logger)

        result = checker._check_doi_consistency(_ibrnet_entry())

        assert result is None
        crossref.get_by_doi.assert_not_called()


class TestCrossrefGetByDoi:
    """The new CrossrefClient.get_by_doi REST helper."""

    def test_returns_message_on_200(self, logger):
        http = MagicMock()
        http._request.return_value = MagicMock(
            status_code=200,
            json=lambda: {"message": {"DOI": "10.0/x", "title": ["Hello"]}},
        )
        client = CrossrefClient(http)

        msg = client.get_by_doi("10.0/x")

        assert msg == {"DOI": "10.0/x", "title": ["Hello"]}
        # Hits the REST /works/{doi} endpoint, not doi.org.
        called_url = http._request.call_args[0][1]
        assert "api.crossref.org/works/" in called_url

    def test_returns_none_on_non_200(self, logger):
        http = MagicMock()
        http._request.return_value = MagicMock(status_code=404, json=lambda: {})
        client = CrossrefClient(http)

        assert client.get_by_doi("10.0/x") is None

    def test_returns_none_on_exception(self, logger):
        http = MagicMock()
        http._request.side_effect = RuntimeError("boom")
        client = CrossrefClient(http)

        assert client.get_by_doi("10.0/x") is None


@pytest.fixture
def logger():
    return logging.getLogger("test_doi_consistency")


@pytest.fixture
def empty_http():
    """HTTP client whose every request looks like a 404/empty result."""
    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return mock


@pytest.fixture
def dead_sources(empty_http):
    """Crossref/DBLP/S2 clients that return nothing by default."""
    return (
        CrossrefClient(empty_http),
        DBLPClient(empty_http),
        SemanticScholarClient(empty_http),
    )
