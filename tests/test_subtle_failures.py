"""Regression tests for subtle correctness blind spots.

These tests were added TDD-style to expose wrong verdicts that the existing
suite did not catch:

1. Nobiliary-particle surnames ("von Mises", "van den Oord") compared
   asymmetrically between the BibTeX entry and the API record, producing
   spurious AUTHOR_MISMATCH / lowered match scores for real papers.
2. ``extract_arxiv_id_from_text`` matching numbers that merely *look* like a
   modern arXiv ID (DOI fragments, arbitrary ``NNNN.NNNNN`` strings) with an
   impossible month, leading to false preprint detection / wrong lookups.
3. Generic single-word journal aliases ("nature", "science") substring-matching
   distinct sibling journals ("Nature Physics") and collapsing them to the same
   canonical venue, masking a genuine venue mismatch.
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
from bibtex_updater.matching import get_canonical_venue
from bibtex_updater.updater import Resolver
from bibtex_updater.utils import (
    PublishedRecord,
    authors_last_names,
    extract_arxiv_id_from_text,
    last_name_from_person,
)


@pytest.fixture
def checker():
    http = MagicMock()
    http._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return FactChecker(
        CrossrefClient(http),
        DBLPClient(http),
        SemanticScholarClient(http),
        FactCheckerConfig(),
        logging.getLogger("test_subtle"),
    )


# ------------- 1. Nobiliary-particle surnames -------------


class TestParticleSurnames:
    def test_last_name_comma_form_strips_particle(self):
        # "von Mises, Ludwig" -- the family token is "von Mises"; the comparable
        # surname key should reduce to "mises" so it matches "Ludwig von Mises".
        assert last_name_from_person("von Mises, Ludwig") == "mises"

    def test_last_name_van_der_comma_form(self):
        assert last_name_from_person("van der Berg, Jan") == "berg"

    def test_last_name_natural_form_unchanged(self):
        assert last_name_from_person("Ludwig von Mises") == "mises"

    def test_particle_only_surname_not_emptied(self):
        # A surname that is *all* particle tokens must not collapse to "".
        assert last_name_from_person("Van") == "van"

    def test_author_matches_across_particle_forms(self, checker):
        # Entry uses "Given particle Family"; API record stores family="van den Oord".
        entry = {
            "ID": "oord2016",
            "ENTRYTYPE": "article",
            "title": "WaveNet: A Generative Model for Raw Audio",
            "author": "Aaron van den Oord and Sander Dieleman",
            "year": "2016",
        }
        record = PublishedRecord(
            doi="10.0000/x",
            title="WaveNet: A Generative Model for Raw Audio",
            authors=[
                {"given": "Aaron", "family": "van den Oord"},
                {"given": "Sander", "family": "Dieleman"},
            ],
            year=2016,
        )
        comparisons = checker._compare_all_fields(entry, record)
        assert comparisons["author"].matches, comparisons["author"].similarity_score


# ------------- 2. arXiv ID over-matching -------------


class TestArxivIdMonthValidation:
    def test_doi_fragment_not_extracted(self):
        # "5678.9012" has an impossible month (78) -> not a real arXiv ID.
        assert extract_arxiv_id_from_text("10.1234/5678.9012") is None

    def test_impossible_month_rejected(self):
        assert extract_arxiv_id_from_text("3499.12345") is None

    def test_valid_id_still_extracted(self):
        assert extract_arxiv_id_from_text("2602.01031") == "2602.01031"

    def test_arxiv_doi_still_extracted(self):
        assert extract_arxiv_id_from_text("10.48550/arXiv.2406.14302") == "2406.14302"

    def test_first_valid_id_when_invalid_precedes(self):
        # An invalid-month number followed by a real arXiv ID -> return the real one.
        assert extract_arxiv_id_from_text("ref 9999.0000 see arXiv:2602.01031") == "2602.01031"

    def test_entry_eprint_impossible_month_not_used(self, checker):
        entry = {"eprint": "5678.9012"}
        assert checker._arxiv_id_from_entry(entry) is None

    def test_legacy_abs_url_not_truncated(self):
        # "arxiv.org/abs/hep-th/9901001" must keep the full legacy ID, not "hep-th".
        assert extract_arxiv_id_from_text("https://arxiv.org/abs/hep-th/9901001") == "hep-th/9901001"

    def test_pdf_url_strips_suffix_and_version(self):
        assert extract_arxiv_id_from_text("https://arxiv.org/pdf/2602.01031v2.pdf") == "2602.01031"

    def test_new_abs_url_unchanged(self):
        assert extract_arxiv_id_from_text("https://arxiv.org/abs/2602.01031") == "2602.01031"


# ------------- 3. Generic single-word venue collapsing -------------


class TestVenueCollapsing:
    def test_nature_sibling_not_collapsed_to_nature(self):
        # "Nature Physics" is a distinct journal from "Nature".
        assert get_canonical_venue("Nature Physics") != "nature"

    def test_science_sibling_not_collapsed(self):
        assert get_canonical_venue("Science Robotics") != "science"

    def test_exact_nature_still_canonicalizes(self):
        assert get_canonical_venue("Nature") == "nature"

    def test_exact_science_still_canonicalizes(self):
        assert get_canonical_venue("Science") == "science"

    def test_acronym_with_short_suffix_still_canonicalizes(self):
        # A single-token acronym with a short trailing word is the *same* venue
        # (unlike a generic-word sibling journal) and must still canonicalize --
        # restricting exact-match to generic journal words must not regress this.
        assert get_canonical_venue("NeurIPS Track") == "neurips"
        assert get_canonical_venue("ICML Conf") == "icml"

    def test_generic_journal_siblings_not_collapsed(self):
        # Sibling journals sharing a generic-word prefix must NOT collapse.
        assert get_canonical_venue("Nature Methods") != "nature"
        assert get_canonical_venue("Science Advances") != "science"
        assert get_canonical_venue("PNAS Nexus") != "pnas"


# ------------- 4. Symmetric surnames in the resolver match score -------------


class TestResolverSurnameSymmetry:
    """The resolver's _compute_match_score must normalize the record-side family
    names the same way as the entry side (authors_last_names). Otherwise a
    particle surname stored comma-first in the .bib (`van den Oord, Aaron`)
    yields entry key `oord` but record key `van den oord`, Jaccard 0, and the
    arXiv-keyed stages / _verify_arxiv_match wrongly reject a correct record.
    """

    @pytest.fixture
    def resolver(self):
        http = MagicMock()
        http._request.return_value = MagicMock(status_code=404, json=lambda: {})
        return Resolver(http=http, logger=logging.getLogger("test_resolver_sym"), scholarly_client=None)

    def test_particle_surname_match_score_symmetric(self, resolver):
        from bibtex_updater.utils import normalize_title_for_match

        title = "WaveNet: A Generative Model for Raw Audio"
        authors_ref = authors_last_names("van den Oord, Aaron")
        rec = PublishedRecord(
            doi="10.0/x",
            title=title,
            authors=[{"given": "Aaron", "family": "van den Oord"}],
            year=2016,
            type="journal-article",
        )
        score = resolver._compute_match_score(normalize_title_for_match(title), rec, authors_ref)
        assert score >= resolver.MATCH_THRESHOLD, score

    def test_verify_arxiv_match_accepts_particle_author(self, resolver):
        from bibtex_updater.utils import normalize_title_for_match

        title = "WaveNet: A Generative Model for Raw Audio"
        entry = {"title": title, "author": "van den Oord, Aaron"}
        rec = PublishedRecord(
            doi="10.0/x",
            title=title,
            authors=[{"given": "Aaron", "family": "van den Oord"}],
            year=2016,
            type="journal-article",
        )
        assert resolver._verify_arxiv_match(rec, entry, normalize_title_for_match(title)) is rec
