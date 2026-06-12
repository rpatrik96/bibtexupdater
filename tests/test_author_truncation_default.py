"""Default-mode silent author-list truncation (HALLMARK ``partial_author_list``).

The benchmark's truncation corruption keeps the FIRST and LAST author and drops
MIDDLE co-authors, so the entry side is an in-order SUBSEQUENCE of the canonical
list -- ``symmetric_author_match`` returns PARTIAL and the entry used to abstain
as UNCONFIRMED (detected at only 16-22%). Default mode now escalates that shape
to AUTHOR_TRUNCATED, but only under tight FPR gates:

  a. PARTIAL with a strictly shorter (sentinel-free) entry author list;
  b. no disclosed truncation ("..."/"et al." in note/howpublished/title, or an
     "and others"/"et al" sentinel in the author field itself);
  c. the best record is order_reliable AND structured_names;
  d. dropped >= 2 OR (>= 3 canonical authors AND dropped >= a third of them);
  e. >= 2 order-reliable sources independently list more authors than the entry.

Strict mode keeps its existing, looser rule 5 behaviour (covered by
tests/test_strict_mode.py).
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
from bibtex_updater.matching import MatchOutcome
from bibtex_updater.utils import PublishedRecord


def _make_checker(strict: bool = False) -> FactChecker:
    fake_http = MagicMock()
    fake_http._request.return_value = MagicMock(status_code=404, json=lambda: {})
    crossref = CrossrefClient(fake_http)
    dblp = DBLPClient(fake_http)
    s2 = SemanticScholarClient(fake_http)
    cfg = FactCheckerConfig(strict=strict)
    return FactChecker(crossref, dblp, s2, cfg, logging.getLogger("truncation-test"))


@pytest.fixture
def default_checker() -> FactChecker:
    return _make_checker(strict=False)


TITLE = "Big Multi-Author Collaboration Paper"

FIVE_AUTHORS = [
    {"given": "Alice", "family": "First"},
    {"given": "Bob", "family": "Second"},
    {"given": "Cathy", "family": "Third"},
    {"given": "Dan", "family": "Fourth"},
    {"given": "Eve", "family": "Last"},
]


def _record(
    authors: list[dict[str, str]],
    order_reliable: bool = True,
    structured: bool = True,
) -> PublishedRecord:
    return PublishedRecord(
        doi="10.1/x",
        title=TITLE,
        authors=authors,
        journal="ICML",
        year=2023,
        order_reliable=order_reliable,
        structured_names=structured,
    )


def _entry(author: str, **extra: str) -> dict[str, str]:
    e = {"title": TITLE, "author": author, "booktitle": "ICML", "year": "2023"}
    e.update(extra)
    return e


class TestDefaultModeSilentTruncation:
    """Benchmark-shaped positive cases: first+last subsequence, corroborated."""

    def test_first_and_last_of_five_flags_author_truncated(self, default_checker):
        # Entry keeps first + last of a 5-author paper (the benchmark's
        # partial_author_list shape), two order-reliable structured sources
        # corroborate the full list -> AUTHOR_TRUNCATED.
        entry = _entry("First, Alice and Last, Eve")
        best = _record(FIVE_AUTHORS)
        per_source = {"crossref": best, "openalex": _record(FIVE_AUTHORS)}
        comps = default_checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.MISMATCH
        assert "Silent author-list truncation" in (comps["author"].note or "")
        status = default_checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.AUTHOR_TRUNCATED

    def test_two_of_three_first_and_last_flags(self, default_checker):
        # 1-of-3 dropped satisfies the ratio arm of gate d (1 >= 3/3).
        three = FIVE_AUTHORS[:2] + [FIVE_AUTHORS[-1]]
        entry = _entry("First, Alice and Last, Eve")
        best = _record(three)
        per_source = {"crossref": best, "openalex": _record(three)}
        comps = default_checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.MISMATCH
        status = default_checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.AUTHOR_TRUNCATED

    def test_first_and_middle_and_last_subsequence_flags(self, default_checker):
        # The generator sometimes keeps one random middle author: first + one
        # middle + last of 5 (dropped=2) is still an in-order subsequence.
        entry = _entry("First, Alice and Third, Cathy and Last, Eve")
        best = _record(FIVE_AUTHORS)
        per_source = {"crossref": best, "openalex": _record(FIVE_AUTHORS)}
        comps = default_checker._compare_all_fields(entry, best, per_source_records=per_source)
        status = default_checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.AUTHOR_TRUNCATED


class TestDefaultModeTruncationFprGates:
    """Each gate independently suppresses the flag -> abstention (UNCONFIRMED)."""

    def test_and_others_leading_prefix_still_matches(self, default_checker):
        # Disclosed truncation via the structured sentinel on a leading prefix
        # is a positive confirmation in the matcher (existing behaviour).
        entry = _entry("First, Alice and Second, Bob and others")
        best = _record(FIVE_AUTHORS)
        per_source = {"crossref": best, "openalex": _record(FIVE_AUTHORS)}
        comps = default_checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comps["author"].matches is True
        status = default_checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.VERIFIED

    def test_and_others_interior_subsequence_stays_unconfirmed(self, default_checker):
        # First+last plus "and others": the matcher keeps the interior elision
        # PARTIAL, and the author-field sentinel counts as DISCLOSED truncation
        # for the new default gate -> no flag, abstain as UNCONFIRMED.
        entry = _entry("First, Alice and Last, Eve and others")
        best = _record(FIVE_AUTHORS)
        per_source = {"crossref": best, "openalex": _record(FIVE_AUTHORS)}
        comps = default_checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        status = default_checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.UNCONFIRMED

    def test_et_al_in_note_field_suppresses_flag(self, default_checker):
        # has_explicit_truncation_indicator covers the note field -> gate b
        # blocks the escalation, the PARTIAL flows to UNCONFIRMED.
        entry = _entry("First, Alice and Last, Eve", note="et al.")
        best = _record(FIVE_AUTHORS)
        per_source = {"crossref": best, "openalex": _record(FIVE_AUTHORS)}
        comps = default_checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        status = default_checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.UNCONFIRMED

    def test_single_full_source_with_stub_second_source_abstains(self, default_checker):
        # Corroboration gate e: only ONE source carries the full list; the
        # other returned a 2-author stub (== entry length, not fuller) -> no
        # flag, abstain.
        entry = _entry("First, Alice and Last, Eve")
        best = _record(FIVE_AUTHORS)
        stub = _record([FIVE_AUTHORS[0], FIVE_AUTHORS[-1]])
        per_source = {"crossref": best, "openalex": stub}
        comps = default_checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        status = default_checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.UNCONFIRMED

    def test_no_per_source_records_abstains(self, default_checker):
        # Without per_source_records there is no corroboration -> abstain.
        entry = _entry("First, Alice and Last, Eve")
        comps = default_checker._compare_all_fields(entry, _record(FIVE_AUTHORS))
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.UNCONFIRMED

    def test_seven_of_eight_authors_abstains(self, default_checker):
        # dropped=1 of 8: neither >= 2 nor >= 8/3 -> a single missing co-author
        # on a long list is a transcription slip, not deliberate truncation.
        eight = FIVE_AUTHORS + [
            {"given": "Frank", "family": "Sixth"},
            {"given": "Grace", "family": "Seventh"},
            {"given": "Henry", "family": "Eighth"},
        ]
        cited = eight[:7]  # leading 7 of 8 -> PARTIAL (prefix without sentinel)
        author_field = " and ".join(f"{a['family']}, {a['given']}" for a in cited)
        entry = _entry(author_field)
        best = _record(eight)
        per_source = {"crossref": best, "openalex": _record(eight)}
        comps = default_checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        status = default_checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.UNCONFIRMED

    def test_unstructured_or_order_unreliable_best_record_abstains(self, default_checker):
        # Gate c: an S2-style flat-name record (not structured, not order
        # reliable) is not an authoritative list -> no flag even when two
        # other sources would corroborate.
        entry = _entry("First, Alice and Last, Eve")
        flat_best = _record(FIVE_AUTHORS, order_reliable=False, structured=False)
        per_source = {
            "crossref": _record(FIVE_AUTHORS),
            "openalex": _record(FIVE_AUTHORS),
        }
        comps = default_checker._compare_all_fields(entry, flat_best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        status = default_checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.UNCONFIRMED

    def test_structured_but_order_unreliable_best_record_abstains(self, default_checker):
        # Gate c requires BOTH order_reliable and structured_names on the best.
        entry = _entry("First, Alice and Last, Eve")
        best = _record(FIVE_AUTHORS, order_reliable=False, structured=True)
        per_source = {
            "crossref": _record(FIVE_AUTHORS),
            "openalex": _record(FIVE_AUTHORS),
        }
        comps = default_checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        status = default_checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.UNCONFIRMED

    def test_order_unreliable_sources_do_not_corroborate(self, default_checker):
        # Gate e counts ORDER-RELIABLE sources only: two S2-style full lists
        # plus the structured best record's own source is just one vote.
        entry = _entry("First, Alice and Last, Eve")
        best = _record(FIVE_AUTHORS)
        per_source = {
            "crossref": best,
            "s2": _record(FIVE_AUTHORS, order_reliable=False, structured=False),
            "arxiv": _record(FIVE_AUTHORS, order_reliable=False, structured=False),
        }
        comps = default_checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        status = default_checker._determine_status(0.95, comps, ["crossref", "s2", "arxiv"])
        assert status is FactCheckStatus.UNCONFIRMED


class TestStrictModeUnchanged:
    """Strict rule 5 keeps its existing, looser behaviour: no corroboration or
    structured/order-reliable requirement (full coverage in test_strict_mode.py)."""

    def test_strict_flags_without_corroboration_or_structured_record(self):
        strict_checker = _make_checker(strict=True)
        entry = _entry("First, Alice and Second, Bob")
        # Plain record: not order_reliable, not structured -- strict still flags.
        record = PublishedRecord(doi="10.1/x", title=TITLE, authors=FIVE_AUTHORS, journal="ICML", year=2023)
        comps = strict_checker._compare_all_fields(entry, record)
        assert comps["author"].outcome is MatchOutcome.MISMATCH
        status = strict_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.AUTHOR_TRUNCATED


class TestAuthorTruncatedCalibration:
    def test_author_truncated_is_a_soft_problem(self):
        from bibtex_updater.calibration import STATUS_BASE_CONFIDENCE

        conf = STATUS_BASE_CONFIDENCE["author_truncated"]
        # Soft-problem tier: same prior as the other single-field mismatches,
        # within the confident-bucket anchor (>= 0.72) and below the strong tier.
        assert conf == STATUS_BASE_CONFIDENCE["author_mismatch"]
        assert 0.72 <= conf < STATUS_BASE_CONFIDENCE["hallucinated"]
