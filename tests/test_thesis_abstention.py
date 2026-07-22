"""Theses are not indexed by paper databases, so a miss is not evidence.

``@phdthesis``/``@mastersthesis`` are absent from Crossref, OpenAlex, DBLP and
Semantic Scholar by construction -- dissertations live in national and
institutional repositories. Routing them through the paper cascade therefore
cannot confirm them, and the top-scoring candidate is whatever unrelated paper
shares a surname: the Varga run bound a Hungarian dissertation on packet-network
QoS to *"Faue, Elizabeth (2017) Rethinking the American Labor Movement"* and
reported its title, author and year as mismatches.

Reporting a mismatch against a record that cannot be the cited work asserts
something the tool has no evidence for. When the title does not confirm, a
thesis abstains (UNCONFIRMED, "could not verify") instead.

This never suppresses positive evidence: ``_validate_year``, ``_validate_doi``,
``_check_arxiv_id_consistency`` and ``_detect_chimeric_title`` all return before
``_determine_status`` is reached, so a fabricated DOI or an impossible year on a
thesis still flags.
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
    PublishedRecord,
    SemanticScholarClient,
)

PROBLEM_STATUSES = {
    FactCheckStatus.TITLE_MISMATCH,
    FactCheckStatus.AUTHOR_MISMATCH,
    FactCheckStatus.YEAR_MISMATCH,
    FactCheckStatus.VENUE_MISMATCH,
    FactCheckStatus.PARTIAL_MATCH,
    FactCheckStatus.GIVEN_NAME_SUBSTITUTION,
}


@pytest.fixture
def checker() -> FactChecker:
    fake_http = MagicMock()
    fake_http._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return FactChecker(
        CrossrefClient(fake_http),
        DBLPClient(fake_http),
        SemanticScholarClient(fake_http),
        FactCheckerConfig(),
        logging.getLogger("thesis-abstention-test"),
    )


def _unrelated_record():
    return PublishedRecord(
        doi="10.13001/jwcs.v3i1.6137",
        url=None,
        title="Faue, Elizabeth (2017) Rethinking the American Labor Movement, Routledge, New York, NY.",
        authors=[{"given": "Joseph", "family": "Varga"}],
        journal="Journal of Working-Class Studies",
        year=2018,
    )


THESIS_ENTRY = {
    "ID": "Varga2011PhD",
    "ENTRYTYPE": "phdthesis",
    "author": "Varga, P{\\'a}l",
    "title": "Csomagkapcsolt h{\\'a}l{\\'o}zatokon ny{\\'u}jtott szolg{\\'a}ltat{\\'a}sok "
    "min{\\H o}s{\\'e}gbiztos{\\'i}t{\\'a}s{\\'a}nak m{\\'o}dszerei {\\'e}s metrik{\\'a}i",
    "school": "Budapest University of Technology and Economics",
    "year": "2011",
}


class TestThesisAbstainsOnUnrelatedRecord:
    def test_phdthesis_does_not_flag_against_unrelated_paper(self, checker):
        record = _unrelated_record()
        comps = checker._compare_all_fields(THESIS_ENTRY, record)
        status = checker._determine_status(0.58, comps, ["crossref"], entry_type="phdthesis")
        assert status not in PROBLEM_STATUSES

    def test_phdthesis_abstains_as_unconfirmed(self, checker):
        record = _unrelated_record()
        comps = checker._compare_all_fields(THESIS_ENTRY, record)
        status = checker._determine_status(0.58, comps, ["crossref"], entry_type="phdthesis")
        assert status is FactCheckStatus.UNCONFIRMED

    def test_mastersthesis_behaves_the_same(self, checker):
        entry = dict(THESIS_ENTRY, ENTRYTYPE="mastersthesis")
        comps = checker._compare_all_fields(entry, _unrelated_record())
        status = checker._determine_status(0.58, comps, ["crossref"], entry_type="mastersthesis")
        assert status not in PROBLEM_STATUSES


class TestNonThesisTypesUnaffected:
    """The gate is scoped to theses; every other type keeps its verdict."""

    def test_article_against_unrelated_record_still_flags(self, checker):
        entry = dict(THESIS_ENTRY, ENTRYTYPE="article", journal="Infocommunications Journal")
        comps = checker._compare_all_fields(entry, _unrelated_record())
        status = checker._determine_status(0.58, comps, ["crossref"], entry_type="article")
        assert status in PROBLEM_STATUSES

    def test_inproceedings_against_unrelated_record_still_flags(self, checker):
        entry = dict(THESIS_ENTRY, ENTRYTYPE="inproceedings", booktitle="Some Conference")
        comps = checker._compare_all_fields(entry, _unrelated_record())
        status = checker._determine_status(0.58, comps, ["crossref"], entry_type="inproceedings")
        assert status in PROBLEM_STATUSES

    def test_default_entry_type_preserves_legacy_behaviour(self, checker):
        """Existing 3-argument callers must be unaffected."""
        comps = checker._compare_all_fields(dict(THESIS_ENTRY, ENTRYTYPE="article"), _unrelated_record())
        assert checker._determine_status(0.58, comps, ["crossref"]) in PROBLEM_STATUSES


class TestThesisStillVerifiesWhenIndexed:
    """A thesis that IS indexed must still verify normally."""

    def test_matching_record_verifies(self, checker):

        entry = {
            "ID": "t",
            "ENTRYTYPE": "phdthesis",
            "author": "Varga, P{\\'a}l",
            "title": "Methods and Metrics of Quality Assurance",
            "year": "2011",
        }
        record = PublishedRecord(
            doi="10.1000/x",
            url=None,
            title="Methods and Metrics of Quality Assurance",
            authors=[{"given": "Pál", "family": "Varga"}],
            journal="",
            year=2011,
        )
        comps = checker._compare_all_fields(entry, record)
        status = checker._determine_status(0.98, comps, ["crossref"], entry_type="phdthesis")
        assert status not in PROBLEM_STATUSES
