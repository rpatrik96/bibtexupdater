"""Tests for arXiv ID extraction from DataCite DOIs (FIX X2).

HALLMARK 2026-synthetic batches carry arXiv DOIs of the form
``10.48550/arXiv.YYMM.NNNNN(vN)?`` as the sole identifier. Before the fix
``_arxiv_id_from_entry`` mined ``eprint``, ``url``, ``howpublished``,
``journal``, ``note`` but NOT ``doi``, so ``_check_arxiv_id_consistency``
(and its downstream ``_id_anchored_author_mismatch``) never fired for
these entries.

The fix adds a final regex over ``entry["doi"]`` matching the DataCite
prefix, case-insensitively, with the version suffix stripped before
``is_valid_arxiv_id`` validation. Negative cases preserve the existing
contract: non-arXiv DOIs and malformed strings return ``None``.
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
    SemanticScholarClient,
)


@pytest.fixture
def logger():
    return logging.getLogger("test_arxiv_datacite_doi")


@pytest.fixture
def empty_http():
    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return mock


@pytest.fixture
def checker(empty_http, logger):
    crossref = CrossrefClient(empty_http)
    dblp = DBLPClient(empty_http)
    s2 = SemanticScholarClient(empty_http)
    return FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)


class TestArxivIdFromDatacitedDoi:
    """``_arxiv_id_from_entry`` must extract the bare arXiv ID from a
    ``10.48550/arXiv.<id>`` DOI. Without this, the 2026-synthetic batch's
    entries (DOI-only, no eprint/url) miss the arXiv-consistency check
    entirely.
    """

    def test_real_arxiv_datacite_doi_extracts_id(self, checker):
        """Real Improved-DDPM DOI -> ``2102.09672``."""
        entry = {
            "ID": "improved_ddpm",
            "ENTRYTYPE": "article",
            "title": "Improved Denoising Diffusion Probabilistic Models",
            "author": "Nichol, Alex and Dhariwal, Prafulla",
            "doi": "10.48550/arXiv.2102.09672",
            "year": "2021",
        }
        assert checker._arxiv_id_from_entry(entry) == "2102.09672"

    def test_versioned_datacite_doi_strips_version(self, checker):
        """``v1`` suffix is stripped before validation; the bare ID returns."""
        entry = {
            "ID": "synthetic2602",
            "ENTRYTYPE": "article",
            "title": "Some Synthetic 2026 Paper",
            "author": "Synth, A.",
            "doi": "10.48550/arXiv.2602.12172v1",
            "year": "2026",
        }
        assert checker._arxiv_id_from_entry(entry) == "2602.12172"

    def test_lowercase_arxiv_in_doi_matches(self, checker):
        """The DataCite prefix is case-insensitive; ``arxiv`` lowercase still
        extracts the bare ID."""
        entry = {
            "ID": "case_insensitive",
            "ENTRYTYPE": "article",
            "title": "A Paper",
            "author": "Author, A.",
            "doi": "10.48550/arxiv.2102.09672",
            "year": "2021",
        }
        assert checker._arxiv_id_from_entry(entry) == "2102.09672"

    def test_non_arxiv_doi_returns_none(self, checker):
        """A regular publisher DOI (IEEE, ACM, Springer, ...) is not an arXiv
        DataCite DOI -> returns ``None`` (unchanged behavior)."""
        entry = {
            "ID": "ieee_paper",
            "ENTRYTYPE": "inproceedings",
            "title": "Some Real Paper",
            "author": "Author, A.",
            "doi": "10.1109/CVPR46437.2021.00469",
            "year": "2021",
        }
        assert checker._arxiv_id_from_entry(entry) is None

    def test_malformed_arxiv_datacite_doi_returns_none(self, checker):
        """A DataCite DOI whose suffix is not a valid arXiv ID shape doesn't
        match the regex -> ``None``."""
        entry = {
            "ID": "malformed",
            "ENTRYTYPE": "article",
            "title": "X",
            "author": "Y, Z",
            "doi": "10.48550/arxiv.abc",
            "year": "2021",
        }
        assert checker._arxiv_id_from_entry(entry) is None

    def test_empty_doi_returns_none(self, checker):
        """Empty / missing DOI -> ``None`` (no field to mine)."""
        entry = {
            "ID": "no_doi",
            "ENTRYTYPE": "article",
            "title": "X",
            "author": "Y, Z",
            "year": "2021",
        }
        assert checker._arxiv_id_from_entry(entry) is None

    def test_eprint_still_takes_precedence(self, checker):
        """An explicit ``eprint`` is the most authoritative arXiv identifier
        on the entry; it must continue to win even when a (possibly
        contradictory) DataCite DOI is also present."""
        entry = {
            "ID": "both_eprint_and_doi",
            "ENTRYTYPE": "article",
            "title": "X",
            "author": "Y, Z",
            "eprint": "2102.09672",
            "archivePrefix": "arXiv",
            "doi": "10.48550/arXiv.1909.12345",
            "year": "2021",
        }
        # ``eprint`` (mined first) wins.
        assert checker._arxiv_id_from_entry(entry) == "2102.09672"
