"""Tests for preprint-as-published detection (HALLMARK ``preprint_as_published``).

The benchmark's offenders cite an arXiv-only paper as if published at a venue
and carry NO doi/eprint, so the legacy ``_check_preprint_status`` (which builds
its S2 lookup id from the entry's own identifiers) never fired. Two new
identifier-less signals close the gap:

1. Matched-record pivot: derive the arXiv ID from the BEST MATCH (its
   converter-stamped ``arxiv_id`` or its own DataCite arXiv DOI), gated on the
   entry-vs-record title similarity, then run the same S2 arXiv-only check.
2. DBLP-CoRR-only: when every strong-title DBLP candidate is the
   ``journals/corr`` stream, the claimed venue canonicalizes to a covered CS
   conference, no source confirms the claim, DBLP did not error, and the entry
   year is at most last year -> PREPRINT_ONLY without any S2 call.

All fakes, no network. Also covers the ``PublishedRecord.arxiv_id`` population
in the converters and the ``arxiv_id_from_datacite_doi`` helper.
"""

from __future__ import annotations

import datetime
import logging
from unittest.mock import MagicMock

import pytest

from bibtex_updater.fact_checker import (
    FactChecker,
    FactCheckerConfig,
    FactCheckStatus,
    FieldComparison,
)
from bibtex_updater.matching import MatchOutcome
from bibtex_updater.sources import openalex_work_to_candidate_record
from bibtex_updater.utils import (
    PublishedRecord,
    arxiv_atom_to_record,
    arxiv_id_from_datacite_doi,
    s2_data_to_record,
)

CURRENT_YEAR = datetime.datetime.now().year

# ------------- Fixtures / helpers -------------


@pytest.fixture
def logger():
    return logging.getLogger("test_preprint_as_published")


def _build_checker(s2_get_paper=None, dblp_hits=(), s2_items=(), cr_items=(), or_notes=()):
    """FactChecker wired to MagicMock clients (no network)."""
    crossref = MagicMock()
    crossref.search.return_value = list(cr_items)
    crossref.http = MagicMock()
    dblp = MagicMock()
    dblp.search.return_value = list(dblp_hits)
    s2 = MagicMock()
    s2.search.return_value = list(s2_items)
    s2.get_paper.return_value = s2_get_paper
    openalex = MagicMock()
    openalex.search.return_value = []
    openreview = MagicMock()
    openreview.search.return_value = list(or_notes)
    checker = FactChecker(
        crossref,
        dblp,
        s2,
        FactCheckerConfig(top_k=3),
        logging.getLogger("test_preprint_as_published"),
        openalex=openalex,
        openreview=openreview,
    )
    return checker, s2


def _entry(**overrides):
    """Identifier-less entry claiming a published venue (the offender shape)."""
    entry = {
        "ID": "widget2023",
        "ENTRYTYPE": "inproceedings",
        "title": "Robust Widget Learning",
        "author": "Lovelace, Ada and Babbage, Charles",
        "booktitle": "NeurIPS",
        "year": "2023",
    }
    entry.update(overrides)
    return entry


def _twin_record(**overrides):
    """The arXiv preprint twin of ``_entry`` as a matched candidate record."""
    kwargs = {
        "doi": "",
        "title": "Robust Widget Learning",
        "authors": [
            {"given": "Ada", "family": "Lovelace"},
            {"given": "Charles", "family": "Babbage"},
        ],
        "journal": None,
        "year": 2023,
        "type": "preprint",
        "arxiv_id": "2301.00001",
    }
    kwargs.update(overrides)
    return PublishedRecord(**kwargs)


def _s2_arxiv_only():
    """S2 paper payload for an arXiv-only paper (no DOI, no venue)."""
    return {
        "title": "Robust Widget Learning",
        "authors": [{"name": "Ada Lovelace"}],
        "venue": "",
        "year": 2023,
        "publicationTypes": [],
        "externalIds": {"ArXiv": "2301.00001"},
        "publicationVenue": None,
    }


def _s2_published():
    """S2 paper payload showing a DOI + venue (published)."""
    return {
        "title": "Robust Widget Learning",
        "authors": [{"name": "Ada Lovelace"}],
        "venue": "NeurIPS",
        "year": 2023,
        "publicationTypes": ["Conference"],
        "externalIds": {"ArXiv": "2301.00001", "DOI": "10.1234/widget"},
        "publicationVenue": {"name": "NeurIPS"},
    }


def _corr_hit(title="Robust Widget Learning", key="journals/corr/abs-2301-00001", venue="CoRR", doi=None):
    """DBLP search hit; defaults to the CoRR (arXiv) stream record."""
    info = {
        "title": title,
        "authors": {"author": [{"text": "Ada Lovelace"}, {"text": "Charles Babbage"}]},
        "venue": venue,
        "year": "2023",
        "key": key,
        "type": "Informal Publications",
    }
    if doi:
        info["doi"] = doi
    return {"info": info}


def _dblp_rec(**overrides):
    """A DBLP-shaped candidate record (CoRR stream by default)."""
    kwargs = {
        "doi": "",
        "title": "Robust Widget Learning",
        "authors": [
            {"given": "Ada", "family": "Lovelace"},
            {"given": "Charles", "family": "Babbage"},
        ],
        "journal": "CoRR",
        "year": 2023,
        "order_reliable": True,
        "venue_key": "journals/corr",
    }
    kwargs.update(overrides)
    return PublishedRecord(**kwargs)


# ===========================================================================
# Matched-record pivot (entry has no doi/eprint)
# ===========================================================================


class TestMatchedRecordPivot:
    def test_pivot_flags_preprint_only_via_twin_arxiv_id(self):
        # (1) No identifiers on the entry; the best match carries the arXiv
        # identity; S2 says arXiv-only -> PREPRINT_ONLY.
        checker, s2 = _build_checker(s2_get_paper=_s2_arxiv_only())
        result = checker._check_preprint_status(_entry(), _twin_record())
        assert result is FactCheckStatus.PREPRINT_ONLY
        s2.get_paper.assert_called_once_with("ARXIV:2301.00001")

    def test_pivot_derives_id_from_datacite_doi(self):
        # The twin may carry only its DataCite arXiv DOI (e.g. a Crossref/
        # OpenAlex-shaped record without the stamped arxiv_id).
        checker, s2 = _build_checker(s2_get_paper=_s2_arxiv_only())
        twin = _twin_record(arxiv_id=None, doi="10.48550/arxiv.2301.00001")
        result = checker._check_preprint_status(_entry(), twin)
        assert result is FactCheckStatus.PREPRINT_ONLY
        s2.get_paper.assert_called_once_with("ARXIV:2301.00001")

    def test_no_flag_when_s2_shows_published(self):
        # (2) S2 shows DOI + venue -> the paper IS published; no flag (the
        # caller keeps its UNCONFIRMED abstention).
        checker, _ = _build_checker(s2_get_paper=_s2_published())
        assert checker._check_preprint_status(_entry(), _twin_record()) is None

    def test_no_flag_when_s2_errors(self):
        # (3) S2 abstains (errors map to None inside get_paper) -> no flag.
        checker, _ = _build_checker(s2_get_paper=None)
        assert checker._check_preprint_status(_entry(), _twin_record()) is None

    def test_pivot_requires_strong_title_match(self):
        # The best match must genuinely be the cited paper's preprint twin.
        checker, s2 = _build_checker(s2_get_paper=_s2_arxiv_only())
        unrelated_twin = _twin_record(title="A Completely Different Paper About Kernels")
        assert checker._check_preprint_status(_entry(), unrelated_twin) is None
        s2.get_paper.assert_not_called()

    def test_pivot_skipped_when_record_has_no_arxiv_identity(self):
        checker, s2 = _build_checker(s2_get_paper=_s2_arxiv_only())
        no_id_twin = _twin_record(arxiv_id=None, doi="")
        assert checker._check_preprint_status(_entry(), no_id_twin) is None
        s2.get_paper.assert_not_called()

    def test_entry_identifier_path_unchanged(self):
        # Legacy semantics: an entry WITH an eprint keeps using its own id.
        checker, s2 = _build_checker(s2_get_paper=_s2_arxiv_only())
        entry = _entry(eprint="2301.00001")
        result = checker._check_preprint_status(entry, _twin_record(arxiv_id="9999.99999"))
        assert result is FactCheckStatus.PREPRINT_ONLY
        s2.get_paper.assert_called_once_with("ARXIV:2301.00001")

    def test_end_to_end_identifierless_entry_flags_preprint_only(self):
        # Full check_entry flow: cascade returns only the S2 arXiv twin
        # (venue-less, carries externalIds.ArXiv) -> UNCONFIRMED would be the
        # old verdict; the pivot upgrades it to PREPRINT_ONLY.
        s2_item = {
            "title": "Robust Widget Learning",
            "authors": [{"name": "Ada Lovelace"}, {"name": "Charles Babbage"}],
            "venue": "",
            "year": 2023,
            "publicationTypes": [],
            "externalIds": {"ArXiv": "2301.00001"},
        }
        checker, s2 = _build_checker(s2_get_paper=_s2_arxiv_only(), s2_items=[s2_item])
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.PREPRINT_ONLY
        s2.get_paper.assert_called_once_with("ARXIV:2301.00001")


# ===========================================================================
# DBLP-CoRR-only signal (no S2 required)
# ===========================================================================


class TestDblpCorrOnlySignal:
    def _call(self, checker, entry, candidates, errors=None, comparisons=None):
        return checker._dblp_corr_only_preprint(
            entry,
            entry.get("booktitle") or entry.get("journal") or "",
            candidates,
            errors or [],
            comparisons,
        )

    def test_corr_only_flags_preprint_only(self):
        # (4) Claimed ICML 2023 (< current year); every strong-title DBLP
        # candidate is journals/corr; no source reports a published venue.
        checker, _ = _build_checker()
        entry = _entry(booktitle="ICML 2023")
        candidates = [(0.95, _dblp_rec(), "dblp")]
        assert self._call(checker, entry, candidates) is FactCheckStatus.PREPRINT_ONLY

    def test_corr_only_sets_venue_note(self):
        checker, _ = _build_checker()
        entry = _entry(booktitle="ICML 2023")
        comparisons = {
            "venue": FieldComparison("venue", "ICML 2023", None, 1.0, False, outcome=MatchOutcome.NON_COMPARABLE)
        }
        candidates = [(0.95, _dblp_rec(), "dblp")]
        status = self._call(checker, entry, candidates, comparisons=comparisons)
        assert status is FactCheckStatus.PREPRINT_ONLY
        assert "only as CoRR" in (comparisons["venue"].note or "")

    def test_no_flag_when_real_proceedings_record_exists(self):
        # (5) DBLP also knows the real conf/icml proceedings record.
        checker, _ = _build_checker()
        entry = _entry(booktitle="ICML 2023")
        proceedings = _dblp_rec(journal="ICML", venue_key="conf/icml")
        candidates = [(0.95, _dblp_rec(), "dblp"), (0.95, proceedings, "dblp")]
        assert self._call(checker, entry, candidates) is None

    def test_no_flag_when_strong_dblp_hit_is_noncorr_other_venue(self):
        # A strong-title DBLP hit indexed at ANY non-CoRR stream (even a
        # different venue) breaks the "DBLP knows it only as CoRR" premise.
        checker, _ = _build_checker()
        entry = _entry(booktitle="ICML 2023")
        aaai = _dblp_rec(journal="AAAI", venue_key="conf/aaai")
        candidates = [(0.95, _dblp_rec(), "dblp"), (0.95, aaai, "dblp")]
        assert self._call(checker, entry, candidates) is None

    def test_no_flag_for_current_year_claims(self):
        # (6) DBLP proceedings indexing lags; current-year claims never flag.
        checker, _ = _build_checker()
        entry = _entry(booktitle=f"ICML {CURRENT_YEAR}", year=str(CURRENT_YEAR))
        candidates = [(0.95, _dblp_rec(year=CURRENT_YEAR), "dblp")]
        assert self._call(checker, entry, candidates) is None

    def test_no_flag_for_journal_venues(self):
        # (7) The DBLP proceedings index is not exhaustive for journals.
        checker, _ = _build_checker()
        entry = _entry(ENTRYTYPE="article", booktitle=None, journal="JMLR")
        del entry["booktitle"]
        candidates = [(0.95, _dblp_rec(), "dblp")]
        assert self._call(checker, entry, candidates) is None

    def test_no_flag_for_uncanonicalizable_venue(self):
        checker, _ = _build_checker()
        entry = _entry(booktitle="Workshop on Imaginary Things")
        candidates = [(0.95, _dblp_rec(), "dblp")]
        assert self._call(checker, entry, candidates) is None

    def test_no_flag_when_dblp_errored(self):
        # (8) A failed DBLP query means its silence is meaningless.
        checker, _ = _build_checker()
        entry = _entry(booktitle="ICML 2023")
        candidates = [(0.95, _dblp_rec(), "dblp")]
        assert self._call(checker, entry, candidates, errors=["DBLP: timeout"]) is None

    def test_no_flag_when_another_source_confirms_claimed_venue(self):
        # Any source reporting a non-preprint venue matching the claim vetoes.
        checker, _ = _build_checker()
        entry = _entry(booktitle="ICML 2023")
        openreview_rec = _dblp_rec(journal="ICML 2023 poster", venue_key=None)
        candidates = [(0.95, _dblp_rec(), "dblp"), (0.95, openreview_rec, "openreview")]
        assert self._call(checker, entry, candidates) is None

    def test_no_flag_on_weak_title_corr_hits(self):
        # A CoRR hit below the 0.95 strong-title bar contributes nothing.
        checker, _ = _build_checker()
        entry = _entry(booktitle="ICML 2023")
        weak = _dblp_rec(title="Robust Widget Learning with Extra Trailing Qualifications Everywhere")
        assert self._call(checker, entry, [(0.7, weak, "dblp")]) is None

    def test_no_flag_without_candidates_or_year(self):
        checker, _ = _build_checker()
        entry = _entry(booktitle="ICML 2023")
        assert self._call(checker, entry, None) is None
        assert self._call(checker, entry, []) is None
        no_year = _entry(booktitle="ICML 2023", year="n.d.")
        assert self._call(checker, no_year, [(0.95, _dblp_rec(), "dblp")]) is None

    def test_end_to_end_corr_only_flags_preprint_only(self):
        # Full check_entry flow: DBLP returns only the CoRR twin (with its
        # DataCite arXiv DOI), S2 abstains -> the CoRR-only signal fires.
        checker, s2 = _build_checker(
            s2_get_paper=None,
            dblp_hits=[_corr_hit(doi="10.48550/ARXIV.2301.00001")],
        )
        result = checker.check_entry(_entry(booktitle="ICML 2023"))
        assert result.status is FactCheckStatus.PREPRINT_ONLY
        assert "only as CoRR" in (result.field_comparisons["venue"].note or "")

    def test_end_to_end_real_proceedings_stays_verified(self):
        # (9) VERIFIED leak guard: a fully-confirmed entry (venue MATCH against
        # the conf/icml proceedings record) must keep VERIFIED even with a
        # poisoned S2 (arXiv-only answer) and a CoRR twin in the pool. The
        # venue-confirmed gate keeps the identifier-less signals out entirely:
        # S2 is never even asked.
        proceedings_hit = {
            "info": {
                "title": "Robust Widget Learning",
                "authors": {"author": [{"text": "Ada Lovelace"}, {"text": "Charles Babbage"}]},
                "venue": "ICML",
                "year": "2023",
                "key": "conf/icml/Lovelace23",
                "type": "Conference and Workshop Papers",
                "doi": "10.5555/icml.2023.42",
            }
        }
        checker, s2 = _build_checker(
            s2_get_paper=_s2_arxiv_only(),
            dblp_hits=[proceedings_hit, _corr_hit(doi="10.48550/ARXIV.2301.00001")],
        )
        result = checker.check_entry(_entry(booktitle="ICML 2023"))
        assert result.status is FactCheckStatus.VERIFIED
        s2.get_paper.assert_not_called()

    def test_check_preprint_status_skips_new_signals_on_confirmed_venue(self):
        # Direct-call variant of (9): with a positively-MATCHed venue the
        # pivot and the CoRR check never run.
        checker, s2 = _build_checker(s2_get_paper=_s2_arxiv_only())
        comparisons = {"venue": FieldComparison("venue", "ICML 2023", "ICML", 0.95, True)}
        candidates = [(0.95, _dblp_rec(), "dblp")]
        entry = _entry(booktitle="ICML 2023")
        status = checker._check_preprint_status(
            entry,
            _twin_record(),
            candidates=candidates,
            errors=[],
            field_comparisons=comparisons,
        )
        assert status is None
        s2.get_paper.assert_not_called()


# ===========================================================================
# PublishedRecord.arxiv_id population (converters + helper)
# ===========================================================================


class TestArxivIdFromDataciteDoi:
    @pytest.mark.parametrize(
        "doi,expected",
        [
            ("10.48550/arXiv.2301.00001", "2301.00001"),
            ("10.48550/ARXIV.2301.00001v2", "2301.00001"),
            ("10.48550/arxiv.2310.12345", "2310.12345"),
            # Impossible month -> not an arXiv ID.
            ("10.48550/arXiv.5678.9012", None),
            # Non-arXiv DOIs untouched.
            ("10.1234/jmlr.2023.001", None),
            ("", None),
            (None, None),
        ],
    )
    def test_extraction(self, doi, expected):
        assert arxiv_id_from_datacite_doi(doi) == expected


class TestConvertersStampArxivId:
    def test_default_is_none(self):
        assert PublishedRecord(doi="10.1/x").arxiv_id is None

    def test_arxiv_atom_to_record(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2301.00001v2</id>
            <title>Robust Widget Learning</title>
            <author><name>Ada Lovelace</name></author>
            <published>2023-01-01T00:00:00Z</published>
          </entry>
        </feed>"""
        rec = arxiv_atom_to_record(xml)
        assert rec is not None
        assert rec.arxiv_id == "2301.00001"  # version stripped

    def test_s2_external_ids(self):
        rec = s2_data_to_record(
            {
                "title": "T",
                "authors": [{"name": "Ada Lovelace"}],
                "venue": "",
                "externalIds": {"ArXiv": "2301.00001v3"},
            }
        )
        assert rec.arxiv_id == "2301.00001"

    def test_s2_datacite_doi_fallback(self):
        rec = s2_data_to_record(
            {
                "title": "T",
                "authors": [],
                "externalIds": {"DOI": "10.48550/arXiv.2301.00002"},
            }
        )
        assert rec.arxiv_id == "2301.00002"

    def test_s2_no_arxiv_identity(self):
        rec = s2_data_to_record({"title": "T", "authors": [], "externalIds": {"DOI": "10.1234/x"}})
        assert rec.arxiv_id is None

    def test_openalex_datacite_doi(self):
        work = {
            "title": "T",
            "doi": "https://doi.org/10.48550/arXiv.2301.00003",
            "authorships": [],
        }
        rec = openalex_work_to_candidate_record(work)
        assert rec.arxiv_id == "2301.00003"

    def test_openalex_primary_location_landing_page(self):
        work = {
            "title": "T",
            "doi": None,
            "authorships": [],
            "primary_location": {
                "landing_page_url": "https://arxiv.org/abs/2301.00004v1",
                "source": {"display_name": "arXiv (Cornell University)"},
            },
        }
        rec = openalex_work_to_candidate_record(work)
        assert rec.arxiv_id == "2301.00004"

    def test_openalex_locations_pdf_url(self):
        work = {
            "title": "T",
            "doi": None,
            "authorships": [],
            "primary_location": {"landing_page_url": "https://example.org/paper"},
            "locations": [
                {"landing_page_url": "https://publisher.example/landing"},
                {"pdf_url": "https://arxiv.org/pdf/2301.00005"},
            ],
        }
        rec = openalex_work_to_candidate_record(work)
        assert rec.arxiv_id == "2301.00005"

    def test_openalex_no_arxiv_identity(self):
        work = {
            "title": "T",
            "doi": "https://doi.org/10.1234/x",
            "authorships": [],
            "primary_location": {"landing_page_url": "https://publisher.example/landing"},
        }
        rec = openalex_work_to_candidate_record(work)
        assert rec.arxiv_id is None

    def test_openalex_defensive_on_malformed_locations(self):
        work = {
            "title": "T",
            "doi": None,
            "authorships": [],
            "primary_location": "not-a-dict",
            "locations": "not-a-list",
        }
        rec = openalex_work_to_candidate_record(work)
        assert rec is not None
        assert rec.arxiv_id is None
