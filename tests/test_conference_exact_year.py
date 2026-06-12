"""Tests for the conference exact-year rule (Task 3).

The default ±1 year tolerance exists for preprint-vs-publication and journal
online-first drift; CONFERENCE proceedings years are exact. HALLMARK's
``arxiv_version_mismatch`` type (real paper, wrong venue OR year ±1) slipped
through the tolerance whenever the venue matched. Default mode now escalates a
±1 year difference to MISMATCH when:

* entry venue and record venue BOTH canonicalize to the SAME conference
  acronym (journal canonicals -- JMLR/TMLR/TPAMI/... -- are exempt);
* the matched record is not a preprint;
* >= 2 order-reliable, non-preprint per-source records agree on the record
  year, so a lone mis-dated deposit never mints the flag.

Strict mode (tolerance 0) is untouched; everything else keeps the existing
tolerance logic.
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
from bibtex_updater.matching import JOURNAL_CANONICAL_VENUES, MatchOutcome
from bibtex_updater.utils import PublishedRecord

TITLE = "Exactness of Proceedings Years"


def _make_checker(strict: bool = False) -> FactChecker:
    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    crossref = CrossrefClient(mock)
    dblp = DBLPClient(mock)
    s2 = SemanticScholarClient(mock)
    return FactChecker(crossref, dblp, s2, FactCheckerConfig(strict=strict), logging.getLogger("test_conf_year"))


@pytest.fixture
def checker() -> FactChecker:
    return _make_checker(strict=False)


def _record(venue: str, year: int | None, *, order_reliable: bool = True, doi: str = "10.1/x") -> PublishedRecord:
    return PublishedRecord(
        doi=doi,
        title=TITLE,
        authors=[{"given": "Ada", "family": "Lovelace"}, {"given": "Alan", "family": "Turing"}],
        journal=venue,
        year=year,
        order_reliable=order_reliable,
        structured_names=True,
    )


def _entry(venue: str = "NeurIPS", year: str = "2023") -> dict[str, str]:
    return {
        "ID": "x",
        "ENTRYTYPE": "inproceedings",
        "title": TITLE,
        "author": "Lovelace, Ada and Turing, Alan",
        "booktitle": venue,
        "year": year,
    }


class TestConferenceExactYear:
    def test_same_conference_year_off_by_one_corroborated_flags(self, checker):
        """Spec test 1: entry NeurIPS 2023 vs two order-reliable records
        agreeing on NeurIPS 2022 -> YEAR_MISMATCH (tolerance no longer
        applies to exact proceedings years)."""
        best = _record("NeurIPS", 2022)
        per_source = {
            "crossref": best,
            "openalex": _record("Advances in Neural Information Processing Systems", 2022),
        }
        comps = checker._compare_all_fields(_entry(), best, per_source_records=per_source)
        assert comps["year"].outcome is MatchOutcome.MISMATCH
        assert "Conference proceedings year is exact" in (comps["year"].note or "")
        status = checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.YEAR_MISMATCH

    def test_single_source_year_tolerated_as_today(self, checker):
        """Spec test 2: only one record carries the year -> no corroboration
        -> the existing ±1 tolerance stands."""
        best = _record("NeurIPS", 2022)
        per_source = {
            "crossref": best,
            "openalex": _record("Advances in Neural Information Processing Systems", None),
        }
        comps = checker._compare_all_fields(_entry(), best, per_source_records=per_source)
        assert comps["year"].outcome is MatchOutcome.MATCH
        assert "Tolerance" in (comps["year"].note or "")

    def test_journal_year_drift_tolerated(self, checker):
        """Spec test 3: JMLR (a journal canonical) ±1 stays tolerated --
        online-first vs issue drift is legitimate for journals."""
        best = _record("Journal of Machine Learning Research", 2022)
        per_source = {
            "crossref": best,
            "openalex": _record("JMLR", 2022),
        }
        entry = _entry(venue="JMLR")
        del entry["booktitle"]
        entry["journal"] = "JMLR"
        comps = checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comps["year"].outcome is MatchOutcome.MATCH

    def test_preprint_best_record_unchanged(self, checker):
        """Spec test 4: a preprint best record keeps today's behaviour -- the
        rule never escalates it (its venue cannot canonicalize and
        record_is_preprint guards it), and beyond tolerance it remains
        NON_COMPARABLE."""
        # Within tolerance: stays MATCH (the twin's year-1 is normal drift).
        best = _record("arXiv preprint arXiv:2301.00001", 2022)
        per_source = {"crossref": best, "openalex": _record("arXiv preprint arXiv:2301.00001", 2022)}
        comps = checker._compare_all_fields(_entry(), best, per_source_records=per_source)
        assert comps["year"].outcome is MatchOutcome.MATCH
        # Beyond tolerance: NON_COMPARABLE exactly as today.
        best_far = _record("arXiv preprint arXiv:2301.00001", 2020)
        comps_far = checker._compare_all_fields(_entry(), best_far, per_source_records={"crossref": best_far})
        assert comps_far["year"].outcome is MatchOutcome.NON_COMPARABLE

    def test_preprint_per_source_records_do_not_corroborate(self, checker):
        """Preprint/CoRR per-source records never count toward the >= 2
        corroboration floor even when they carry the same year."""
        best = _record("NeurIPS", 2022)
        per_source = {
            "crossref": best,
            "semanticscholar": _record("arXiv preprint arXiv:2301.00001", 2022),
            "openalex": _record("NeurIPS", 2022, doi="10.48550/arXiv.2301.00001"),
        }
        comps = checker._compare_all_fields(_entry(), best, per_source_records=per_source)
        # crossref is the only valid corroborator (S2 record is preprint-venued,
        # the openalex one carries a preprint DOI) -> below the floor -> tolerated.
        assert comps["year"].outcome is MatchOutcome.MATCH

    def test_order_unreliable_records_do_not_corroborate(self, checker):
        best = _record("NeurIPS", 2022)
        per_source = {
            "crossref": best,
            "semanticscholar": _record("NeurIPS", 2022, order_reliable=False),
        }
        comps = checker._compare_all_fields(_entry(), best, per_source_records=per_source)
        assert comps["year"].outcome is MatchOutcome.MATCH

    def test_different_canonical_venues_keep_tolerance(self, checker):
        """The rule needs the SAME canonical conference on both sides; a
        wrong-venue shape is the venue checks' territory, not the year's."""
        best = _record("ICML", 2022)
        per_source = {"crossref": best, "openalex": _record("ICML", 2022)}
        comps = checker._compare_all_fields(_entry(venue="NeurIPS"), best, per_source_records=per_source)
        assert comps["year"].outcome is not MatchOutcome.MISMATCH

    def test_sources_disagreeing_on_year_do_not_flag(self, checker):
        """Corroboration means agreeing on the RECORD year: a 2-source split
        (2022 vs 2024) leaves only one corroborator -> tolerated."""
        best = _record("NeurIPS", 2022)
        per_source = {
            "crossref": best,
            "openalex": _record("NeurIPS", 2024),
        }
        comps = checker._compare_all_fields(_entry(), best, per_source_records=per_source)
        assert comps["year"].outcome is MatchOutcome.MATCH

    def test_exact_year_match_untouched(self, checker):
        best = _record("NeurIPS", 2023)
        per_source = {"crossref": best, "openalex": _record("NeurIPS", 2023)}
        comps = checker._compare_all_fields(_entry(), best, per_source_records=per_source)
        assert comps["year"].outcome is MatchOutcome.MATCH

    def test_strict_mode_unchanged(self):
        """Strict mode already runs tolerance 0; the rule must not interfere
        (its note stays the strict tolerance note, not the conference rule's)."""
        strict_checker = _make_checker(strict=True)
        best = _record("NeurIPS", 2022)
        per_source = {"crossref": best, "openalex": _record("NeurIPS", 2022)}
        comps = strict_checker._compare_all_fields(_entry(), best, per_source_records=per_source)
        assert comps["year"].outcome is MatchOutcome.MISMATCH
        assert "Tolerance: ±0" in (comps["year"].note or "")

    def test_no_per_source_records_keeps_tolerance(self, checker):
        best = _record("NeurIPS", 2022)
        comps = checker._compare_all_fields(_entry(), best)
        assert comps["year"].outcome is MatchOutcome.MATCH

    def test_journal_frozenset_contents(self):
        """The exemption list pins the journal canonicals named in the spec."""
        assert {
            "jmlr",
            "tmlr",
            "tpami",
            "ijcv",
            "tacl",
            "nature",
            "science",
            "pnas",
            "nature_mi",
            "nature_comm",
        } == set(JOURNAL_CANONICAL_VENUES)
