"""Tests for the --strict evaluation mode (arXiv 2026 hallucination policy).

Strict mode raises the bar in five asymmetric-cost places (a leaked hallucinated
reference is far worse than an FP):

1. Title Levenshtein-1 near-miss (TITLE_NEAR_MISS).
2. Year tolerance 0 (preprint-twin routes to STRICT_WARN_PREPRINT_YEAR).
3. Author-set: single-source, single-extra-author flag (was 2/2).
4. Author order: alphabetization escape DISABLED -- same-multiset reordering
   against an order-reliable source is a real MISMATCH.
5. Partial author list: silent leading-prefix/subsequence WITHOUT sentinel ->
   AUTHOR_TRUNCATED. With sentinel (or explicit truncation indicator) it stays
   MATCH.

Plus --strict-warn-cnv (requires --strict): promotes NOT_FOUND/UNCONFIRMED to
STRICT_WARN_CNV so CI can fail on exhaustive review.

Every test asserts BOTH the positive case (strict flags) AND the negative case
(default does NOT flag, OR strict respects the disclosed-truncation exception).
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
from bibtex_updater.matching import MatchOutcome, symmetric_author_match
from bibtex_updater.utils import PublishedRecord

# ------------- fixtures -------------


@pytest.fixture
def logger():
    return logging.getLogger("test_strict_mode")


@pytest.fixture
def fake_http():
    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return mock


def _make_checker(strict: bool = False, strict_warn_cnv: bool = False) -> FactChecker:
    fake_http = MagicMock()
    fake_http._request.return_value = MagicMock(status_code=404, json=lambda: {})
    crossref = CrossrefClient(fake_http)
    dblp = DBLPClient(fake_http)
    s2 = SemanticScholarClient(fake_http)
    cfg = FactCheckerConfig(strict=strict, strict_warn_cnv=strict_warn_cnv)
    return FactChecker(crossref, dblp, s2, cfg, logging.getLogger("strict-test"))


@pytest.fixture
def strict_checker() -> FactChecker:
    return _make_checker(strict=True)


@pytest.fixture
def default_checker() -> FactChecker:
    return _make_checker(strict=False)


# ------------- 1. Title Levenshtein-1 near-miss -------------


class TestStrictTitleLev1NearMiss:
    def test_strict_flags_lev1_near_miss(self, strict_checker):
        """'Subspace Differential Privacys' vs 'Subspace Differential Privacy'
        (one trailing 's') -> TITLE_NEAR_MISS in strict mode.
        """
        entry = {
            "title": "Subspace Differential Privacys",
            "author": "Smith, John",
            "year": "2020",
        }
        record = PublishedRecord(
            doi="10.1/x",
            title="Subspace Differential Privacy",
            authors=[{"given": "John", "family": "Smith"}],
            year=2020,
        )
        comparisons = strict_checker._compare_all_fields(entry, record)
        title = comparisons["title"]
        # The near-miss is a MISMATCH in strict mode (not a confirmed MATCH).
        # ``outcome`` is None so the default ``resolved_outcome`` derives from
        # ``matches=False`` -> MISMATCH.
        assert title.is_mismatch is True
        assert title.matches is False
        assert "Strict near-miss" in (title.note or "")
        # And the gate routes the lone title MISMATCH to TITLE_NEAR_MISS.
        status = strict_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status is FactCheckStatus.TITLE_NEAR_MISS

    def test_default_mode_does_not_flag_lev1(self, default_checker):
        """The same Lev-1 difference is a MATCH in default mode (fuzzy score
        is already at the threshold; only the strict gate fires)."""
        entry = {
            "title": "Subspace Differential Privacys",
            "author": "Smith, John",
            "year": "2020",
        }
        record = PublishedRecord(
            doi="10.1/x",
            title="Subspace Differential Privacy",
            authors=[{"given": "John", "family": "Smith"}],
            year=2020,
        )
        comparisons = default_checker._compare_all_fields(entry, record)
        title = comparisons["title"]
        # Default: above fuzzy threshold, no Lev-1 escalation -> MATCH.
        assert title.matches is True
        assert title.outcome is None or title.outcome is MatchOutcome.MATCH

    def test_diacritic_variant_not_flagged_in_strict(self, strict_checker):
        """Negative: a legitimate diacritic variant ('Reseau' vs 'Réseau')
        normalizes to edit distance 0 (handled by ``normalize_title_for_match``'s
        diacritic fold), so it does NOT trip the Lev-1 gate in strict mode."""
        entry = {"title": "Reseau", "author": "Smith, John", "year": "2020"}
        record = PublishedRecord(
            doi="10.1/x",
            title="Réseau",
            authors=[{"given": "John", "family": "Smith"}],
            year=2020,
        )
        comparisons = strict_checker._compare_all_fields(entry, record)
        title = comparisons["title"]
        # Diacritic fold collapses the difference -> MATCH, no near-miss.
        assert title.matches is True


# ------------- 2. Year tolerance 0 + preprint-twin abstain -------------


class TestStrictYearTolerance:
    def test_strict_flags_one_year_diff(self, strict_checker):
        """entry=2022 record=2023 -> default tolerates (default tol=1), strict
        flags YEAR_MISMATCH (tol=0). Non-preprint record so the preprint-twin
        abstain path does NOT fire."""
        entry = {"title": "Test", "author": "Smith, John", "year": "2022"}
        record = PublishedRecord(
            doi="10.1/x",
            title="Test",
            authors=[{"given": "John", "family": "Smith"}],
            journal="ICML",
            year=2023,
        )
        comparisons = strict_checker._compare_all_fields(entry, record)
        assert comparisons["year"].outcome is MatchOutcome.MISMATCH
        # Strict-mode lone year mismatch -> YEAR_MISMATCH (positive evidence).
        status = strict_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status is FactCheckStatus.YEAR_MISMATCH

    def test_default_tolerates_one_year_diff(self, default_checker):
        """Default tolerance 1 -> the same entry/record pair is a MATCH."""
        entry = {"title": "Test", "author": "Smith, John", "year": "2022"}
        record = PublishedRecord(
            doi="10.1/x",
            title="Test",
            authors=[{"given": "John", "family": "Smith"}],
            journal="ICML",
            year=2023,
        )
        comparisons = default_checker._compare_all_fields(entry, record)
        assert comparisons["year"].matches is True

    def test_strict_routes_preprint_twin_year_to_warn(self, strict_checker):
        """Preprint twin (arXiv DOI) with year drift -> STRICT_WARN_PREPRINT_YEAR
        (abstain), not YEAR_MISMATCH. The user is told "year cannot be anchored"."""
        entry = {"title": "Test", "author": "Smith, John", "year": "2022"}
        record = PublishedRecord(
            doi="10.48550/arXiv.1910.03834",  # arXiv DOI -> preprint twin
            title="Test",
            authors=[{"given": "John", "family": "Smith"}],
            year=2019,
        )
        comparisons = strict_checker._compare_all_fields(entry, record)
        # The year field abstains, never reads as a mismatch.
        assert comparisons["year"].outcome is MatchOutcome.NON_COMPARABLE
        status = strict_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status is FactCheckStatus.STRICT_WARN_PREPRINT_YEAR


# ------------- 3. Author-set: single-source single-extra-author -------------


class TestStrictAuthorFabricationThresholds:
    def test_strict_flags_single_source_single_extra(self, strict_checker):
        """Entry adds ONE surname not in the single canonical record's full
        list -> strict-flags AUTHOR_MISMATCH (single source, single extra).
        Default mode requires 2+ sources AND 2+ absent surnames."""
        entry = {
            "title": "Some Real Paper",
            "author": "Real, Alice and Real, Bob and Fabricated, Cathy",
            "year": "2023",
        }
        # Single order-reliable record (Crossref) with only the two real authors.
        canonical = PublishedRecord(
            doi="10.1/x",
            title="Some Real Paper",
            authors=[
                {"given": "Alice", "family": "Real"},
                {"given": "Bob", "family": "Real"},
            ],
            journal="ICML",
            year=2023,
            order_reliable=True,
        )
        per_source = {"crossref": canonical}
        comparisons = strict_checker._compare_all_fields(entry, canonical, per_source_records=per_source)
        # Strict detection escalates to MISMATCH on the single-source single-extra.
        assert comparisons["author"].outcome is MatchOutcome.MISMATCH
        assert "fabricated" in (comparisons["author"].note or "").lower()

    def test_default_does_not_flag_single_source_single_extra(self, default_checker):
        """Default thresholds (2 sources, 2 absent surnames) -> no flag, the
        author check passes through as a PARTIAL (entry longer than record)."""
        entry = {
            "title": "Some Real Paper",
            "author": "Real, Alice and Real, Bob and Fabricated, Cathy",
            "year": "2023",
        }
        canonical = PublishedRecord(
            doi="10.1/x",
            title="Some Real Paper",
            authors=[
                {"given": "Alice", "family": "Real"},
                {"given": "Bob", "family": "Real"},
            ],
            journal="ICML",
            year=2023,
            order_reliable=True,
        )
        per_source = {"crossref": canonical}
        comparisons = default_checker._compare_all_fields(entry, canonical, per_source_records=per_source)
        # Default: PARTIAL (consistent-but-incomplete on the record side), not
        # escalated to MISMATCH.
        assert comparisons["author"].outcome is not MatchOutcome.MISMATCH

    def test_sentinel_still_suppresses_in_strict(self, strict_checker):
        """Even in strict mode an 'and others' sentinel suppresses the
        fabrication check -- the citation discloses the truncation."""
        entry = {
            "title": "Some Real Paper",
            "author": "Real, Alice and Real, Bob and others",
            "year": "2023",
        }
        canonical = PublishedRecord(
            doi="10.1/x",
            title="Some Real Paper",
            authors=[
                {"given": "Alice", "family": "Real"},
                {"given": "Bob", "family": "Real"},
            ],
            journal="ICML",
            year=2023,
            order_reliable=True,
        )
        per_source = {"crossref": canonical}
        comparisons = strict_checker._compare_all_fields(entry, canonical, per_source_records=per_source)
        # Sentinel: no fabrication flag.
        assert comparisons["author"].outcome is not MatchOutcome.MISMATCH


# ------------- 4. Author order: alphabetization guard disabled in strict -------------


class TestStrictAuthorOrderAlphabetized:
    def test_strict_flags_same_multiset_alphabetized_swap(self):
        """df33d8b shape: entry and record share the SAME author multiset but
        a different lead; the record is alphabetized (record-side sort artifact).
        Default tolerates (looks_alphabetized escape); strict treats it as a
        real swap (MISMATCH)."""
        # Three authors, both lists; record alphabetized; lead author differs.
        entry = ["zhang", "alpha", "beta"]
        api = sorted(entry)  # alphabetized -> ["alpha", "beta", "zhang"]
        # Different lead, same multiset.
        default = symmetric_author_match(entry, api, order_reliable=True, strict=False)
        strict = symmetric_author_match(entry, api, order_reliable=True, strict=True)
        # Default: the alphabetization guard rescues it as a MATCH.
        assert default.outcome is MatchOutcome.MATCH
        # Strict: the same-multiset reordering is a real swap.
        assert strict.outcome is MatchOutcome.MISMATCH

    def test_strict_lead_difference_against_order_reliable_is_mismatch(self):
        """Same-multiset, different lead, order-reliable source: strict flags
        even with no other evidence (alphabetization escape disabled)."""
        entry = ["smith", "alpha", "doe"]
        # api: same multiset but rotated; record-side alphabetized
        api = ["alpha", "doe", "smith"]
        r = symmetric_author_match(entry, api, order_reliable=True, strict=True)
        assert r.outcome is MatchOutcome.MISMATCH


# ------------- 5. Partial author list silent truncation -------------


class TestStrictAuthorTruncation:
    def test_strict_flags_leading_prefix_without_sentinel(self, strict_checker):
        """Entry is a leading-prefix of the canonical with NO sentinel ->
        strict AUTHOR_TRUNCATED. Default lets it through as PARTIAL ->
        UNCONFIRMED (could-not-confirm)."""
        entry = {
            "title": "Big Multi-Author Paper",
            "author": "Real, Alice and Real, Bob",
            "year": "2023",
        }
        canonical = PublishedRecord(
            doi="10.1/x",
            title="Big Multi-Author Paper",
            authors=[
                {"given": "Alice", "family": "Real"},
                {"given": "Bob", "family": "Real"},
                {"given": "Cathy", "family": "Co"},
                {"given": "Dan", "family": "Co"},
            ],
            journal="ICML",
            year=2023,
        )
        comparisons = strict_checker._compare_all_fields(entry, canonical)
        # Strict escalates the silent leading-prefix to MISMATCH, then the
        # gate routes the lone author MISMATCH to AUTHOR_TRUNCATED.
        assert comparisons["author"].outcome is MatchOutcome.MISMATCH
        status = strict_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status is FactCheckStatus.AUTHOR_TRUNCATED

    def test_sentinel_keeps_match_in_strict(self, strict_checker):
        """Same leading-prefix WITH 'and others' sentinel -> still MATCH in
        strict (the citation discloses the truncation)."""
        entry = {
            "title": "Big Multi-Author Paper",
            "author": "Real, Alice and Real, Bob and others",
            "year": "2023",
        }
        canonical = PublishedRecord(
            doi="10.1/x",
            title="Big Multi-Author Paper",
            authors=[
                {"given": "Alice", "family": "Real"},
                {"given": "Bob", "family": "Real"},
                {"given": "Cathy", "family": "Co"},
                {"given": "Dan", "family": "Co"},
            ],
            journal="ICML",
            year=2023,
        )
        comparisons = strict_checker._compare_all_fields(entry, canonical)
        # Sentinel-confirmed truncation -> MATCH, never escalated.
        assert comparisons["author"].matches is True

    def test_explicit_truncation_indicator_in_note_keeps_match(self, strict_checker):
        """Sibling-field truncation indicator ('et al.' in note field) ->
        strict respects the disclosed truncation, no AUTHOR_TRUNCATED escalation."""
        entry = {
            "title": "Big Multi-Author Paper",
            "author": "Real, Alice and Real, Bob",
            "note": "et al.",
            "year": "2023",
        }
        canonical = PublishedRecord(
            doi="10.1/x",
            title="Big Multi-Author Paper",
            authors=[
                {"given": "Alice", "family": "Real"},
                {"given": "Bob", "family": "Real"},
                {"given": "Cathy", "family": "Co"},
                {"given": "Dan", "family": "Co"},
            ],
            journal="ICML",
            year=2023,
        )
        comparisons = strict_checker._compare_all_fields(entry, canonical)
        # The disclosed truncation indicator in `note` keeps the verdict PARTIAL,
        # not escalated to MISMATCH.
        assert comparisons["author"].outcome is not MatchOutcome.MISMATCH


# ------------- 6. --strict-warn-cnv promotion -------------


class TestStrictWarnCnv:
    def test_strict_warn_cnv_promotes_not_found(self):
        checker = _make_checker(strict=True, strict_warn_cnv=True)
        # Promotion goes through the helper.
        promoted = checker._apply_strict_warn_cnv(FactCheckStatus.NOT_FOUND)
        assert promoted is FactCheckStatus.STRICT_WARN_CNV

    def test_strict_warn_cnv_promotes_unconfirmed(self):
        checker = _make_checker(strict=True, strict_warn_cnv=True)
        promoted = checker._apply_strict_warn_cnv(FactCheckStatus.UNCONFIRMED)
        assert promoted is FactCheckStatus.STRICT_WARN_CNV

    def test_strict_without_warn_cnv_keeps_not_found(self):
        """Strict alone (no --strict-warn-cnv) -> NOT_FOUND stays NOT_FOUND."""
        checker = _make_checker(strict=True, strict_warn_cnv=False)
        promoted = checker._apply_strict_warn_cnv(FactCheckStatus.NOT_FOUND)
        assert promoted is FactCheckStatus.NOT_FOUND

    def test_warn_cnv_without_strict_is_inert(self):
        """warn-cnv without strict is inert (the CLI rejects the combination,
        but if a caller bypasses the CLI guard the helper does nothing)."""
        checker = _make_checker(strict=False, strict_warn_cnv=True)
        promoted = checker._apply_strict_warn_cnv(FactCheckStatus.NOT_FOUND)
        assert promoted is FactCheckStatus.NOT_FOUND

    def test_strict_warn_cnv_does_not_touch_problematic(self):
        """Problematic statuses are NOT routed through STRICT_WARN_CNV --
        CNV is a fourth class, distinct from PROBLEMATIC."""
        checker = _make_checker(strict=True, strict_warn_cnv=True)
        for s in (
            FactCheckStatus.AUTHOR_MISMATCH,
            FactCheckStatus.YEAR_MISMATCH,
            FactCheckStatus.HALLUCINATED,
            FactCheckStatus.VERIFIED,
        ):
            assert checker._apply_strict_warn_cnv(s) is s


# ------------- Bucket placement -------------


class TestStrictStatusBuckets:
    def test_strict_warn_preprint_year_is_abstained(self):
        from bibtex_updater.fact_checker import _is_abstained_status

        assert _is_abstained_status(FactCheckStatus.STRICT_WARN_PREPRINT_YEAR) is True

    def test_strict_warn_cnv_is_not_abstained(self):
        from bibtex_updater.fact_checker import _is_abstained_status

        # STRICT_WARN_CNV is the user's opt-in fourth bucket -- distinct from
        # abstained AND problematic so CI can fail on it cleanly.
        assert _is_abstained_status(FactCheckStatus.STRICT_WARN_CNV) is False

    def test_title_near_miss_in_problematic_bucket(self):
        from bibtex_updater.fact_checker import _is_abstained_status

        # Positive evidence under the arXiv 2026 policy -> problematic, not abstained.
        assert _is_abstained_status(FactCheckStatus.TITLE_NEAR_MISS) is False
        assert _is_abstained_status(FactCheckStatus.AUTHOR_TRUNCATED) is False


# ------------- CLI surface -------------


class TestStrictCliSurface:
    def test_strict_flag_parsed(self):
        from bibtex_updater.fact_checker import build_parser

        p = build_parser()
        ns = p.parse_args(["x.bib", "--strict"])
        assert ns.strict is True
        assert ns.strict_warn_cnv is False

    def test_strict_warn_cnv_flag_parsed(self):
        from bibtex_updater.fact_checker import build_parser

        p = build_parser()
        ns = p.parse_args(["x.bib", "--strict", "--strict-warn-cnv"])
        assert ns.strict is True
        assert ns.strict_warn_cnv is True
