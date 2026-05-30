"""Tests for cross-source venue verification (FIX X1, SCoRe-shape leak).

The standard venue block only flags a MISMATCH when the *best* candidate
returns a different canonical venue. When the best-scoring candidate is the
entry's own preprint twin (NON_COMPARABLE) or happens to mirror the entry's
mis-cited venue, a real wrong-venue claim slips through.

Concrete leak case ``cb518c15992d``: an entry claims ``venue=NeurIPS`` for
*Tao Yu, Rui Zhang, Alex Polozov, Christopher Meek, Ahmed Hassan Awadallah --
"SCoRe: Pre-Training for Context Representation in Conversational Semantic
Parsing"*. The real venue is ICLR 2021 (OpenReview ``oyZxhRI2RiE``).

The fix adds ``_detect_cross_source_venue_mismatch`` -- the venue analogue of
``_detect_author_fabrication`` -- which downgrades the venue outcome to
MISMATCH when (a) >= 2 order-reliable sources contributed a published-venue
candidate, (b) those candidates canonicalize to the SAME venue, (c) the
entry's venue is itself canonicalizable, AND (d) the consensus differs from
the entry's canonical. Single dissenter, preprint records, and ambiguous
splits never trip it.
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
from bibtex_updater.matching import MatchOutcome
from bibtex_updater.utils import PublishedRecord


@pytest.fixture
def logger():
    return logging.getLogger("test_cross_source_venue")


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


def _score_entry() -> dict[str, str]:
    """SCoRe-shape entry: claims NeurIPS, real venue is ICLR."""
    return {
        "ID": "score2021",
        "ENTRYTYPE": "inproceedings",
        "title": "SCoRe: Pre-Training for Context Representation in Conversational Semantic Parsing",
        "author": ("Yu, Tao and Zhang, Rui and Polozov, Alex and " "Meek, Christopher and Awadallah, Ahmed Hassan"),
        "booktitle": "NeurIPS",
        "year": "2021",
    }


def _record(title: str, venue: str, *, order_reliable: bool = True) -> PublishedRecord:
    """Minimal order-reliable PublishedRecord with a populated venue."""
    return PublishedRecord(
        doi="10.0/x",
        title=title,
        authors=[
            {"given": "Tao", "family": "Yu"},
            {"given": "Rui", "family": "Zhang"},
            {"given": "Alex", "family": "Polozov"},
            {"given": "Christopher", "family": "Meek"},
            {"given": "Ahmed Hassan", "family": "Awadallah"},
        ],
        journal=venue,
        year=2021,
        order_reliable=order_reliable,
    )


class TestCrossSourceVenueMismatch:
    """The SCoRe-shape leak: NeurIPS-claimed entry where >= 2 sources agree
    on ICLR. The cross-source consensus must downgrade the venue outcome to
    MISMATCH so the verdict surfaces as VENUE_MISMATCH instead of a silent
    VERIFIED / UNCONFIRMED.
    """

    def test_score_shape_two_sources_agree_flags(self, checker):
        """Positive: entry venue NeurIPS, the BEST match is a preprint twin
        (NON_COMPARABLE in the standard venue block), but two order-reliable
        sources agree on ICLR -> the cross-source check downgrades the
        outcome to MISMATCH so VENUE_MISMATCH still surfaces.

        This is the SCoRe-shape leak: a top-scoring arXiv twin returns
        a non-comparable venue and silently lets the wrong-venue claim
        through; the cross-source check restores the flag."""
        entry = _score_entry()
        # Best match: arXiv preprint twin -> venue check returns
        # NON_COMPARABLE on the published venue claim.
        best = _record(entry["title"], "arXiv preprint arXiv:2010.03546")
        per_source = {
            "crossref": _record(entry["title"], "ICLR"),
            "openalex": _record(entry["title"], "ICLR"),
        }
        comparisons = checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comparisons["venue"].outcome is MatchOutcome.MISMATCH
        assert comparisons["venue"].matches is False
        assert "iclr" in (comparisons["venue"].note or "").lower()

    def test_score_shape_two_sources_agree_flags_with_matching_best(self, checker):
        """Positive variant: best match already reports ICLR (standard block
        flags MISMATCH on its own). The cross-source path is a no-op here
        because the venue is already MISMATCH; this confirms the helper
        does not collide with the standard block."""
        entry = _score_entry()
        best = _record(entry["title"], "ICLR")
        per_source = {
            "crossref": _record(entry["title"], "ICLR"),
            "openalex": _record(entry["title"], "ICLR"),
        }
        comparisons = checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comparisons["venue"].outcome is MatchOutcome.MISMATCH
        assert comparisons["venue"].matches is False

    def test_single_dissenter_does_not_flag(self, checker):
        """Negative: one source agrees with the entry (NeurIPS), one disagrees
        (ICLR) -> no consensus, no flag."""
        entry = _score_entry()
        best = _record(entry["title"], "NeurIPS")
        per_source = {
            "crossref": _record(entry["title"], "NeurIPS"),
            "openalex": _record(entry["title"], "ICLR"),
        }
        comparisons = checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comparisons["venue"].outcome is not MatchOutcome.MISMATCH

    def test_preprint_venue_does_not_anchor(self, checker):
        """Negative: a disagreeing source's venue is a preprint/series
        (NON_COMPARABLE) and cannot anchor a published-venue claim. With
        only one valid published-venue source contributing, no consensus
        -> no flag.

        Uses a preprint-twin best_match so the standard venue block returns
        NON_COMPARABLE; the cross-source path is the only thing that could
        flag, and per the gate it abstains (one preprint source + one
        published source is sub-floor)."""
        entry = _score_entry()
        # Preprint-twin best match -> standard block returns NON_COMPARABLE.
        best = _record(entry["title"], "arXiv preprint arXiv:2010.03546")
        per_source = {
            "crossref": _record(entry["title"], "arXiv preprint arXiv:2010.03546"),
            "openalex": _record(entry["title"], "ICLR"),
        }
        comparisons = checker._compare_all_fields(entry, best, per_source_records=per_source)
        # Only 1 order-reliable published-venue contributor (openalex). Below
        # the 2-source consensus floor -> no cross-source flag.
        assert comparisons["venue"].outcome is not MatchOutcome.MISMATCH

    def test_entry_venue_blank_does_not_flag(self, checker):
        """Negative: entry has no venue claim -> nothing to refute, the
        venue field is vacuously confirmed (no claim)."""
        entry = _score_entry()
        del entry["booktitle"]
        best = _record(entry["title"], "ICLR")
        per_source = {
            "crossref": _record(entry["title"], "ICLR"),
            "openalex": _record(entry["title"], "ICLR"),
        }
        comparisons = checker._compare_all_fields(entry, best, per_source_records=per_source)
        # No claim -> MATCH ("No venue claimed"). Cross-source consensus is
        # irrelevant when the entry makes no claim.
        assert comparisons["venue"].outcome is not MatchOutcome.MISMATCH
        assert comparisons["venue"].matches is True

    def test_entry_venue_non_canonical_does_not_flag(self, checker):
        """Negative: the entry's venue does not canonicalize to a recognized
        published venue (random nonsense string) -> we cannot meaningfully
        refute it via cross-source consensus -> no flag."""
        entry = _score_entry()
        entry["booktitle"] = "Some Random Workshop That Does Not Exist"
        best = _record(entry["title"], "ICLR")
        per_source = {
            "crossref": _record(entry["title"], "ICLR"),
            "openalex": _record(entry["title"], "ICLR"),
        }
        _ = checker._compare_all_fields(entry, best, per_source_records=per_source)
        # The entry venue isn't canonicalizable, so the cross-source check
        # abstains. The standard venue block may still register MISMATCH
        # (two real different venues compared fuzzily); the key assertion
        # is that the cross-source path itself does not flag a venue we
        # cannot canonicalize against.
        consensus = checker._detect_cross_source_venue_mismatch(entry["booktitle"], per_source)
        assert consensus is None

    def test_all_sources_agree_with_entry_venue_matches(self, checker):
        """Positive negative-test: when all order-reliable sources agree with
        the entry's canonical venue, the venue check stays MATCH. This
        verifies that a positive cross-source consensus is not collateral
        damage of the new helper."""
        entry = _score_entry()
        entry["booktitle"] = "ICLR"
        best = _record(entry["title"], "ICLR")
        per_source = {
            "crossref": _record(entry["title"], "ICLR"),
            "openalex": _record(entry["title"], "ICLR"),
        }
        comparisons = checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comparisons["venue"].outcome is MatchOutcome.MATCH
        assert comparisons["venue"].matches is True

    def test_order_unreliable_sources_do_not_anchor(self, checker):
        """Negative: only order-unreliable sources (e.g. S2 flat names)
        disagreeing with the entry never trip the cross-source flag --
        the helper mirrors ``_record_full_surname_union``'s order-reliable
        gate. Best match is a preprint twin so the standard venue block
        returns NON_COMPARABLE; the cross-source helper would otherwise
        be the only path that could flag."""
        entry = _score_entry()
        best = _record(entry["title"], "arXiv preprint arXiv:2010.03546")
        per_source = {
            "semanticscholar": _record(entry["title"], "ICLR", order_reliable=False),
        }
        comparisons = checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comparisons["venue"].outcome is not MatchOutcome.MISMATCH
