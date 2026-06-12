"""Tests for the DOI- and arXiv-anchored fast paths.

After the entry's own identifier consistency check finds no mismatch, the
checker may skip the multi-source cascade when the single authoritative record
behind that identifier FULLY confirms every claimed field. The fast paths can
only short-circuit a clean VERIFIED -- every partial/non-comparable/mismatch
outcome falls through to the normal cascade -- and they are inert in --strict
mode and under --no-fast-path.

All fakes; the cascade-skip claims are proven by counting search calls on the
fakes (crossref.search / openalex.search / dblp.search / s2.search).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from bibtex_updater.fact_checker import (
    FactChecker,
    FactCheckerConfig,
    FactCheckStatus,
    build_parser,
)
from bibtex_updater.utils import PublishedRecord

DOI = "10.1109/CVPR46437.2021.00469"
TITLE = "IBRNet: Learning Multi-View Image-Based Rendering"

FIVE_AUTHORS = [
    {"given": "Carlos", "family": "Garcia"},
    {"given": "Mei", "family": "Lin"},
    {"given": "Priya", "family": "Raman"},
    {"given": "Tom", "family": "Becker"},
    {"given": "Sara", "family": "Martinez"},
]


def _crossref_msg(title=TITLE, doi=DOI, authors=None, venue="CVPR", year=2021, typ="proceedings-article"):
    msg = {
        "DOI": doi,
        "type": typ,
        "title": [title],
        "author": authors
        or [
            {"given": "Qianqian", "family": "Wang"},
            {"given": "Kyle", "family": "Genova"},
        ],
        "issued": {"date-parts": [[year]]},
    }
    if venue:
        msg["container-title"] = [venue]
    return msg


def _doi_entry(author="Wang, Qianqian and Genova, Kyle", venue="CVPR", year="2021"):
    entry = {
        "ID": "ibrnet2021",
        "ENTRYTYPE": "inproceedings",
        "title": TITLE,
        "author": author,
        "doi": DOI,
        "year": year,
    }
    if venue:
        entry["booktitle"] = venue
    return entry


def _openalex_work(title=TITLE, authors=None, venue="CVPR", year=2021):
    names = [f"{a['given']} {a['family']}" for a in (authors or FIVE_AUTHORS)]
    return {
        "title": title,
        "doi": f"https://doi.org/{DOI}",
        "authorships": [{"author": {"display_name": n}} for n in names],
        "primary_location": {"source": {"display_name": venue}},
        "publication_year": year,
        "type": "proceedings-article",
    }


def _build_checker(get_by_doi_msg=None, cr_search_items=(), oa_items=(), config=None):
    crossref = MagicMock()
    crossref.get_by_doi.return_value = get_by_doi_msg
    crossref.search.return_value = list(cr_search_items)
    crossref.http = MagicMock()
    crossref.http.s2_api_key = None  # keep the legacy keyless cascade
    dblp = MagicMock()
    dblp.search.return_value = []
    s2 = MagicMock()
    s2.search.return_value = []
    s2.get_paper.return_value = None
    openalex = MagicMock()
    openalex.search.return_value = list(oa_items)
    openreview = MagicMock()
    openreview.search.return_value = []
    fc = FactChecker(
        crossref,
        dblp,
        s2,
        config or FactCheckerConfig(),
        logging.getLogger("test_fast_paths"),
        openalex=openalex,
        openreview=openreview,
    )
    return fc


def _check(fc, entry):
    return fc.check_entry(entry, pre_validated_dois={entry["ID"]: True})


def _cascade_search_calls(fc):
    return (
        fc.crossref.search.call_count
        + fc.openalex.search.call_count
        + fc.dblp.search.call_count
        + fc.s2.search.call_count
    )


# ===========================================================================
# DOI fast path
# ===========================================================================


class TestDoiFastPath:
    def test_fully_confirmed_doi_entry_verified_with_zero_search_calls(self):
        fc = _build_checker(get_by_doi_msg=_crossref_msg())
        result = _check(fc, _doi_entry())

        assert result.status == FactCheckStatus.VERIFIED
        # The whole cascade was skipped: no search anywhere.
        assert fc.crossref.search.call_count == 0
        assert fc.openalex.search.call_count == 0
        assert fc.dblp.search.call_count == 0
        assert fc.s2.search.call_count == 0
        # Reported like the existing DOI-anchored early returns.
        assert result.api_sources_queried == ["crossref"]
        assert result.api_sources_with_hits == ["crossref"]
        assert result.best_match is not None and result.best_match.doi == DOI.lower()
        assert all(c.is_confirmed for c in result.field_comparisons.values())
        # Assembly mirrors the normal path: calibrated + numeric confidences.
        assert result.overall_confidence > 0.0
        assert float(getattr(result, "confidence_score", 0.0)) > 0.0
        # Single source -> no intersection bonus, no suspects.
        assert result.author_intersection is not None
        assert result.author_intersection.bonus == 0.0
        assert result.author_intersection.suspect == []

    def test_no_fast_path_flag_runs_cascade(self):
        cfg = FactCheckerConfig(doi_fast_path=False, arxiv_fast_path=False)
        fc = _build_checker(get_by_doi_msg=_crossref_msg(), cr_search_items=[_crossref_msg()], config=cfg)
        result = _check(fc, _doi_entry())

        assert result.status == FactCheckStatus.VERIFIED  # verdict-neutral
        assert _cascade_search_calls(fc) > 0
        assert fc.crossref.search.call_count >= 1

    def test_strict_mode_disables_fast_path(self):
        cfg = FactCheckerConfig(strict=True)
        assert cfg.doi_fast_path and cfg.arxiv_fast_path  # flags stay True...
        fc = _build_checker(get_by_doi_msg=_crossref_msg(), cr_search_items=[_crossref_msg()], config=cfg)
        result = _check(fc, _doi_entry())

        # ...but the fast path is inert: the cascade runs.
        assert _cascade_search_calls(fc) > 0
        assert result.status == FactCheckStatus.VERIFIED

    def test_hybrid_fabrication_wrong_title_still_doi_mismatch(self):
        # Real DOI resolving to a DIFFERENT paper: caught by the consistency
        # check BEFORE the fast path, with zero cascade calls.
        fc = _build_checker(get_by_doi_msg=_crossref_msg(title="Delving into Localization Errors for Monocular 3D"))
        result = _check(fc, _doi_entry())

        assert result.status == FactCheckStatus.DOI_MISMATCH
        assert _cascade_search_calls(fc) == 0

    def test_hybrid_fabrication_wrong_authors_still_author_mismatch(self):
        # Real DOI + matching title but fabricated authors: the ID-anchored
        # author check fires before the fast path ever runs.
        fc = _build_checker(
            get_by_doi_msg=_crossref_msg(
                authors=[{"given": "Alan", "family": "Turing"}, {"given": "Ada", "family": "Lovelace"}]
            )
        )
        result = _check(fc, _doi_entry())

        assert result.status == FactCheckStatus.AUTHOR_MISMATCH
        assert _cascade_search_calls(fc) == 0

    def test_truncated_author_list_falls_through_and_flags_author_truncated(self):
        # Entry cites 2 of 5 authors (first + last, no "et al."). The fast path
        # sees a PARTIAL author comparison -> falls through; the cascade then
        # corroborates the full list via TWO order-reliable sources and flags
        # AUTHOR_TRUNCATED (the default-mode truncation gate needs >= 2).
        msg5 = _crossref_msg(authors=FIVE_AUTHORS)
        fc = _build_checker(
            get_by_doi_msg=msg5,
            cr_search_items=[msg5],
            oa_items=[_openalex_work(authors=FIVE_AUTHORS)],
        )
        entry = _doi_entry(author="Garcia, Carlos and Martinez, Sara")
        result = _check(fc, entry)

        assert _cascade_search_calls(fc) > 0  # reached the cascade
        assert result.status == FactCheckStatus.AUTHOR_TRUNCATED

    def test_preprint_doi_record_cannot_confirm_venue_claim_falls_through(self):
        # DataCite arXiv DOI: the DOI record is a preprint, so the claimed
        # published venue is NON_COMPARABLE -> never fast-path VERIFIED.
        datacite_doi = "10.48550/arXiv.2301.12345"
        msg = {
            "DOI": datacite_doi,
            "type": "posted-content",
            "title": [TITLE],
            "author": [
                {"given": "Qianqian", "family": "Wang"},
                {"given": "Kyle", "family": "Genova"},
            ],
            "issued": {"date-parts": [[2021]]},
        }
        fc = _build_checker(get_by_doi_msg=msg)
        entry = _doi_entry()
        entry["doi"] = datacite_doi
        result = _check(fc, entry)

        assert _cascade_search_calls(fc) > 0  # fell through to the cascade
        assert result.status != FactCheckStatus.VERIFIED
        # With every search source empty this abstains, never verifies.
        assert result.status == FactCheckStatus.NOT_FOUND

    def test_unconfirmed_venue_falls_through(self):
        # The DOI record carries no venue while the entry claims one: venue is
        # NON_COMPARABLE -> the fast path must NOT mint VERIFIED (nor any
        # abstention) -- the cascade decides.
        fc = _build_checker(get_by_doi_msg=_crossref_msg(venue=None))
        result = _check(fc, _doi_entry(venue="CVPR"))

        assert _cascade_search_calls(fc) > 0
        assert result.status != FactCheckStatus.VERIFIED

    def test_cli_flag_parses(self):
        args = build_parser().parse_args(["refs.bib", "--no-fast-path"])
        assert args.no_fast_path is True
        args_default = build_parser().parse_args(["refs.bib"])
        assert args_default.no_fast_path is False

    def test_config_defaults_enabled(self):
        cfg = FactCheckerConfig()
        assert cfg.doi_fast_path is True
        assert cfg.arxiv_fast_path is True


# ===========================================================================
# arXiv fast path
# ===========================================================================

ARXIV_ID = "2301.12345"
ARXIV_TITLE = "Context-Aware Sparse Deep Coordination Graphs"


def _arxiv_rec(authors=None, year=2023, title=ARXIV_TITLE):
    # Mirrors arxiv_atom_to_record output: synthesized names, order-unreliable.
    return PublishedRecord(
        doi="",
        url=f"http://arxiv.org/abs/{ARXIV_ID}v1",
        title=title,
        authors=authors
        or [
            {"given": "Tonghan", "family": "Wang"},
            {"given": "Liang", "family": "Zeng"},
        ],
        journal=None,
        year=year,
        type="preprint",
        arxiv_id=ARXIV_ID,
    )


def _arxiv_entry(author="Wang, Tonghan and Zeng, Liang", venue=None, year="2023"):
    entry = {
        "ID": "wang2023sparse",
        "ENTRYTYPE": "misc",
        "title": ARXIV_TITLE,
        "author": author,
        "eprint": ARXIV_ID,
        "archiveprefix": "arXiv",
        "year": year,
    }
    if venue:
        entry["booktitle"] = venue
    return entry


def _arxiv_checker(rec, config=None):
    fc = _build_checker(config=config)
    fc.arxiv = MagicMock()  # non-None gates the consistency check + fast path
    fc._arxiv_record_cache[ARXIV_ID] = rec  # memoized record, zero network
    return fc


class TestArxivFastPath:
    def test_venueless_exact_authors_verified_with_zero_search_calls(self):
        fc = _arxiv_checker(_arxiv_rec())
        result = fc.check_entry(_arxiv_entry())

        assert result.status == FactCheckStatus.VERIFIED
        assert _cascade_search_calls(fc) == 0
        assert result.api_sources_queried == ["arxiv"]
        assert result.api_sources_with_hits == ["arxiv"]
        assert result.best_match is not None and result.best_match.arxiv_id == ARXIV_ID
        assert all(c.is_confirmed for c in result.field_comparisons.values())
        assert float(getattr(result, "confidence_score", 0.0)) > 0.0

    def test_transposed_authors_fall_through_to_cascade(self):
        # Same multiset, different order: the exact-sequence gate fails (the
        # normal matcher would wave this through because arXiv records are
        # order_reliable=False), so the cascade runs and order-reliable
        # sources get their chance to catch a real swap.
        fc = _arxiv_checker(_arxiv_rec())
        result = fc.check_entry(_arxiv_entry(author="Zeng, Liang and Wang, Tonghan"))

        assert _cascade_search_calls(fc) > 0
        assert result is not None  # whatever the cascade decides; no fast path

    def test_venue_claiming_entry_falls_through_to_cascade(self):
        # An arXiv record can never confirm a published-venue claim.
        fc = _arxiv_checker(_arxiv_rec())
        result = fc.check_entry(_arxiv_entry(venue="NeurIPS"))

        assert _cascade_search_calls(fc) > 0
        assert result.status != FactCheckStatus.VERIFIED

    def test_entry_with_doi_not_eligible(self):
        fc = _arxiv_checker(_arxiv_rec())
        entry = _arxiv_entry()
        entry["doi"] = DOI
        fc.crossref.get_by_doi.return_value = None  # consistency can't decide
        result = fc.check_entry(entry, pre_validated_dois={entry["ID"]: True})

        assert _cascade_search_calls(fc) > 0
        assert result is not None

    def test_strict_mode_disables_arxiv_fast_path(self):
        fc = _arxiv_checker(_arxiv_rec(), config=FactCheckerConfig(strict=True))
        fc.check_entry(_arxiv_entry())
        assert _cascade_search_calls(fc) > 0

    def test_no_fast_path_config_disables_arxiv_fast_path(self):
        fc = _arxiv_checker(_arxiv_rec(), config=FactCheckerConfig(doi_fast_path=False, arxiv_fast_path=False))
        fc.check_entry(_arxiv_entry())
        assert _cascade_search_calls(fc) > 0

    def test_year_outside_tolerance_falls_through(self):
        fc = _arxiv_checker(_arxiv_rec(year=2020))
        result = fc.check_entry(_arxiv_entry(year="2023"))
        assert _cascade_search_calls(fc) > 0
        assert result.status != FactCheckStatus.VERIFIED

    def test_missing_author_falls_through(self):
        # Entry lists only the first of two authors: not an exact sequence.
        fc = _arxiv_checker(_arxiv_rec())
        result = fc.check_entry(_arxiv_entry(author="Wang, Tonghan"))
        assert _cascade_search_calls(fc) > 0
        assert result.status != FactCheckStatus.VERIFIED


# ===========================================================================
# Fast-path / strict interaction sanity
# ===========================================================================


@pytest.mark.parametrize("strict", [False, True])
def test_fast_path_never_produces_negative_verdict(strict):
    """A non-confirming DOI record must NEVER yield a fast-path verdict: the
    result (whatever it is) must come from the cascade path, proven by the
    cascade fakes having been queried."""
    msg = _crossref_msg(venue="ICLR")  # record venue contradicts the claim
    fc = _build_checker(get_by_doi_msg=msg, config=FactCheckerConfig(strict=strict))
    entry = _doi_entry(venue="CVPR")
    result = _check(fc, entry)

    # The ID-anchored venue mismatch catches this upstream (strict or not);
    # the fast path itself never returned anything for it.
    assert result.status == FactCheckStatus.VENUE_MISMATCH
    assert _cascade_search_calls(fc) == 0
