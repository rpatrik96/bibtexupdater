"""Alphabetized-record / mixed-initial given-name FP gating (HALLMARK).

The benchmark flagged given_name_substitution / author_mismatch on VALID
entries (4 per split). Driving ``_compare_all_fields`` with realistic record
shapes reproduced three distinct mechanisms:

1. ``same_surname_given_order_violation`` anchored a shared-surname RUN-ORDER
   conclusion on a record whose author list is ALPHABETIZED (Crossref
   NeurIPS-2023 deposits, 10.52202 prefix): sorting re-orders the two
   'Nguyen's, so the run order is a record-side artifact, not a swap.
2. ``given_name_position_audit``'s lead-position both-repeat guard escalated
   when ANY same-surname candidate graded as a substitution -- even though
   ANOTHER candidate benignly explained the entry's given ("Quang" is
   explained by "Quang Ho"; the positional "Khoi" is the OTHER Nguyen). And a
   record-side repeated surname in an alphabetized record can cross-pair two
   different same-surname authors at any position.
3. ``classify_given_pair`` graded a MIXED initial+name given ("J. Westerborn"
   vs "Johan") as a full-name SUBSTITUTION: the all-initials branch requires
   every token to be an initial, so the single-letter first token fell through
   to the full-vs-full Levenshtein comparison ("j" vs "johan" = 4 > 3).
4. A record alphabetized by its GIVEN-FIRST display string ("Anh Tuan Tran" <
   "Khoi Nguyen" < ...) is invisible to the matcher's surname-key
   alphabetization escape (keys [tran, nguyen, nguyen, vu] are not A-Z), so a
   same-multiset reordering minted a MISMATCH from a sort artifact.

Fixes (default mode; strict untouched): the run-order check and the audit's
positional anchoring never trust an alphabetized record's order; the lead
guard abstains when ANY same-surname candidate is benign; single-letter first
given tokens compare as initials, never as substitutions; and a >= 3-author
same-multiset MISMATCH against a display-alphabetized record softens to
PARTIAL -> UNCONFIRMED (abstention -- never a positive flag, never VERIFIED
on an order no source can confirm).

Regression fixtures: "Dataset Diffusion" (NeurIPS 2023) and "Pseudo-Marginal
Hamiltonian Monte Carlo" (JMLR 2021) must never produce a problem status
(VERIFIED or UNCONFIRMED acceptable; not AUTHOR_MISMATCH and not
GIVEN_NAME_SUBSTITUTION).
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
from bibtex_updater.utils import (
    GIVEN_VARIETY_CLASS,
    PublishedRecord,
    classify_given_pair,
    given_name_position_audit,
    record_looks_alphabetized,
    same_surname_given_order_violation,
)

PROBLEM_STATUSES = {
    FactCheckStatus.AUTHOR_MISMATCH,
    FactCheckStatus.GIVEN_NAME_SUBSTITUTION,
    FactCheckStatus.PARTIAL_MATCH,
    FactCheckStatus.HALLUCINATED,
}


def _make_checker(strict: bool = False) -> FactChecker:
    fake_http = MagicMock()
    fake_http._request.return_value = MagicMock(status_code=404, json=lambda: {})
    crossref = CrossrefClient(fake_http)
    dblp = DBLPClient(fake_http)
    s2 = SemanticScholarClient(fake_http)
    cfg = FactCheckerConfig(strict=strict)
    return FactChecker(crossref, dblp, s2, cfg, logging.getLogger("alpha-fp-test"))


@pytest.fixture
def default_checker() -> FactChecker:
    return _make_checker(strict=False)


def _record(pairs, order_reliable=True, structured=True, journal="x", year=2023, title="T"):
    return PublishedRecord(
        doi="10.1/x",
        title=title,
        authors=[{"given": g, "family": f} for g, f in pairs],
        journal=journal,
        year=year,
        order_reliable=order_reliable,
        structured_names=structured,
    )


# ---------------------------------------------------------------------------
# record_looks_alphabetized
# ---------------------------------------------------------------------------


class TestRecordLooksAlphabetized:
    def test_surname_sorted_record_is_alphabetized(self):
        rec = _record([("Khoi", "Nguyen"), ("Quang Ho", "Nguyen"), ("Anh Tuan", "Tran"), ("Truong Thanh", "Vu")])
        assert record_looks_alphabetized(rec) is True

    def test_display_string_sorted_record_is_alphabetized(self):
        # Sorted by the given-first display name; surname keys are NOT A-Z.
        rec = _record([("Anh Tuan", "Tran"), ("Khoi", "Nguyen"), ("Quang Ho", "Nguyen"), ("Truong Thanh", "Vu")])
        assert record_looks_alphabetized(rec) is True

    def test_publication_order_record_is_not_alphabetized(self):
        rec = _record([("Quang Ho", "Nguyen"), ("Truong Thanh", "Vu"), ("Anh Tuan", "Tran"), ("Khoi", "Nguyen")])
        assert record_looks_alphabetized(rec) is False

    def test_fewer_than_three_names_never_alphabetized(self):
        # With < 3 names sorted order coincides too often to carry signal.
        rec = _record([("Yadong", "Mu"), ("Zhicheng", "Sun")])
        assert record_looks_alphabetized(rec) is False

    def test_sdedit_publication_order_is_not_alphabetized(self):
        rec = _record(
            [
                ("Chenlin", "Meng"),
                ("Yutong", "He"),
                ("Yang", "Song"),
                ("Jiaming", "Song"),
                ("Jun-Yan", "Zhu"),
            ]
        )
        assert record_looks_alphabetized(rec) is False


# ---------------------------------------------------------------------------
# classify_given_pair: mixed initial+name forms
# ---------------------------------------------------------------------------


class TestClassifyGivenPairMixedInitial:
    def _class(self, e, r):
        return GIVEN_VARIETY_CLASS[classify_given_pair(e, r)]

    def test_matching_mixed_initial_is_benign(self):
        # "J. Westerborn" vs "Johan": the single-letter first token can only
        # testify about the initial -> benign, never a SUBSTITUTION.
        assert self._class("Johan", "J. Westerborn") == "confirmed"
        assert self._class("J. Westerborn", "Johan") == "confirmed"

    def test_conflicting_mixed_initial_softens(self):
        # A conflicting initial is the same low-confidence class as the
        # all-initials branch -- soften, not escalate.
        assert self._class("Johan", "K. Westerborn") == "soften"

    def test_middle_name_delta_still_benign(self):
        assert self._class("Quang", "Quang Ho") == "confirmed"

    def test_full_substitution_still_escalates(self):
        # The mixed-initial fold must not weaken genuine substitutions.
        assert self._class("Yujing", "Yue") == "escalate"
        assert self._class("Quang", "Khoi") == "escalate"


# ---------------------------------------------------------------------------
# same_surname_given_order_violation: alphabetized records never anchor
# ---------------------------------------------------------------------------


class TestSameSurnameOrderViolationAlphabetizedGate:
    DD_ENTRY = "Quang Nguyen and Truong Vu and Anh Tran and Khoi Nguyen"

    def test_alphabetized_record_does_not_anchor_run_order(self):
        # Crossref 10.52202 shape: A-Z by surname re-orders the Nguyen run
        # (Khoi before Quang) -> run-order difference is a sort artifact.
        rec = _record([("Khoi", "Nguyen"), ("Quang Ho", "Nguyen"), ("Anh Tuan", "Tran"), ("Truong Thanh", "Vu")])
        assert same_surname_given_order_violation(self.DD_ENTRY, rec) is False

    def test_publication_order_swap_still_fires(self):
        # The SDEdit-shape true positive (record in publication order, entry
        # swaps the two Songs) must keep firing.
        rec = _record(
            [
                ("Chenlin", "Meng"),
                ("Yutong", "He"),
                ("Yang", "Song"),
                ("Jiaming", "Song"),
                ("Jun-Yan", "Zhu"),
            ]
        )
        entry = "Chenlin Meng and Yutong He and Jiaming Song and Yang Song and Jun-Yan Zhu"
        assert same_surname_given_order_violation(entry, rec) is True


# ---------------------------------------------------------------------------
# given_name_position_audit: alphabetized shared-surname runs + benign lead
# ---------------------------------------------------------------------------


class TestAuditAlphabetizedSharedSurnameGuard:
    DD_ENTRY = "Quang Nguyen and Truong Vu and Anh Tran and Khoi Nguyen"
    ALPHA_SURNAME = [("Khoi", "Nguyen"), ("Quang Ho", "Nguyen"), ("Anh Tuan", "Tran"), ("Truong Thanh", "Vu")]

    def test_benign_candidate_abstains_at_lead(self):
        # Position 0 pairs entry "Quang" with the record's "Khoi" (the OTHER
        # Nguyen, moved there by the A-Z sort) -- but "Quang Ho" benignly
        # explains the entry's given, so the audit must abstain, not escalate.
        worst, findings = given_name_position_audit(self.DD_ENTRY, _record(self.ALPHA_SURNAME))
        assert worst != "escalate"
        assert not any(f["variety"] == "given_name_substitution" for f in findings)

    def test_phantom_same_surname_author_still_escalates(self):
        # An entry given that matches NONE of the record's same-surname authors
        # is a genuine substitution whichever author was meant -> still flags.
        entry = "Phantom Nguyen and Truong Vu and Anh Tran and Khoi Nguyen"
        worst, findings = given_name_position_audit(entry, _record(self.ALPHA_SURNAME))
        assert worst == "escalate"
        assert any(f["variety"] == "given_name_substitution" for f in findings)

    def test_record_side_repeat_in_alphabetized_record_abstains_off_lead(self):
        # Entry cites ONE of the two same-surname authors; the alphabetized
        # record holds the OTHER one at the aligned position. The record-side
        # repeat + alphabetization triggers the candidate-set guard anywhere,
        # not only at the lead.
        entry = "Anh Tran and Quang Nguyen and Khoi Nguyen and Truong Vu"
        rec = _record(self.ALPHA_SURNAME)
        worst, findings = given_name_position_audit(entry, rec)
        assert worst != "escalate"

    def test_d67418_substitution_unchanged(self):
        # The canonical true positive (no repeats, publication-order record)
        # is untouched by the new guards.
        entry = "Durmus Acar and Yujing Zhao and Rafael Navarro and Matthew Mattina"
        rec = _record([("Durmus", "Acar"), ("Yue", "Zhao"), ("Ramon", "Navarro"), ("Matthew", "Mattina")])
        worst, findings = given_name_position_audit(entry, rec)
        assert worst == "escalate"
        assert any(f["variety"] == "given_name_substitution" for f in findings)


# ---------------------------------------------------------------------------
# End-to-end regression fixtures (HALLMARK verified FPs)
# ---------------------------------------------------------------------------


class TestDatasetDiffusionRegression:
    """Valid NeurIPS 2023 entry; Crossref deposits alphabetized."""

    TITLE = "Dataset Diffusion: Diffusion-based Synthetic Data Generation for Pixel-Level Semantic Segmentation"
    VENUE = "Advances in Neural Information Processing Systems"
    ENTRY = {
        "title": TITLE,
        "author": "Quang Nguyen and Truong Vu and Anh Tran and Khoi Nguyen",
        "booktitle": VENUE,
        "year": "2023",
    }

    def _rec(self, pairs, **kw):
        return _record(pairs, journal=self.VENUE, year=2023, title=self.TITLE, **kw)

    PUBLICATION = [("Quang Ho", "Nguyen"), ("Truong Thanh", "Vu"), ("Anh Tuan", "Tran"), ("Khoi", "Nguyen")]
    ALPHA_SURNAME = [("Khoi", "Nguyen"), ("Quang Ho", "Nguyen"), ("Anh Tuan", "Tran"), ("Truong Thanh", "Vu")]
    ALPHA_DISPLAY = [("Anh Tuan", "Tran"), ("Khoi", "Nguyen"), ("Quang Ho", "Nguyen"), ("Truong Thanh", "Vu")]

    @pytest.mark.parametrize("shape", ["PUBLICATION", "ALPHA_SURNAME", "ALPHA_DISPLAY"])
    def test_no_problem_status_for_any_record_shape(self, default_checker, shape):
        record = self._rec(getattr(self, shape))
        comps = default_checker._compare_all_fields(self.ENTRY, record, per_source_records={"crossref": record})
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status not in PROBLEM_STATUSES, f"{shape}: flagged {status}"
        assert status in (FactCheckStatus.VERIFIED, FactCheckStatus.UNCONFIRMED)

    def test_publication_order_record_verifies(self, default_checker):
        record = self._rec(self.PUBLICATION)
        comps = default_checker._compare_all_fields(self.ENTRY, record)
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.VERIFIED

    def test_display_alphabetized_record_abstains(self, default_checker):
        # The given-first display sort is invisible to the matcher's
        # surname-key escape; the call-site gate softens to abstention.
        record = self._rec(self.ALPHA_DISPLAY)
        comps = default_checker._compare_all_fields(self.ENTRY, record)
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.UNCONFIRMED

    def test_strict_mode_unchanged_on_display_alphabetized_record(self):
        # Strict mode does not honour alphabetization escapes at all -- the
        # same-multiset reordering stays a MISMATCH there (existing policy).
        strict_checker = _make_checker(strict=True)
        record = self._rec(self.ALPHA_DISPLAY)
        comps = strict_checker._compare_all_fields(self.ENTRY, record)
        status = strict_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.AUTHOR_MISMATCH


class TestPseudoMarginalHmcRegression:
    """Valid JMLR 2021 entry; diacritics + varied-source given-name forms."""

    TITLE = "Pseudo-Marginal Hamiltonian Monte Carlo"
    VENUE = "Journal of Machine Learning Research"
    ENTRY = {
        "title": TITLE,
        "author": "Johan Alenlöv and Arnoud Doucet and Fredrik Lindsten",
        "journal": VENUE,
        "year": "2021",
    }

    def _rec(self, pairs, **kw):
        return _record(pairs, journal=self.VENUE, year=2021, title=self.TITLE, **kw)

    SHAPES = {
        "canonical": [("Johan", "Alenlöv"), ("Arnaud", "Doucet"), ("Fredrik", "Lindsten")],
        "ascii_folded": [("Johan", "Alenlov"), ("Arnaud", "Doucet"), ("Fredrik", "Lindsten")],
        "mixed_initial_former_name": [
            ("J. Westerborn", "Alenlöv"),
            ("Arnaud", "Doucet"),
            ("Fredrik", "Lindsten"),
        ],
        "initials": [("J.", "Alenlöv"), ("A.", "Doucet"), ("F.", "Lindsten")],
        "given_sorted": [("Arnaud", "Doucet"), ("Fredrik", "Lindsten"), ("Johan", "Alenlöv")],
    }

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    @pytest.mark.parametrize("structured", [True, False])
    def test_no_problem_status_for_any_record_shape(self, default_checker, shape, structured):
        record = self._rec(self.SHAPES[shape], structured=structured)
        comps = default_checker._compare_all_fields(self.ENTRY, record, per_source_records={"crossref": record})
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status not in PROBLEM_STATUSES, f"{shape}/structured={structured}: flagged {status}"
        assert status in (FactCheckStatus.VERIFIED, FactCheckStatus.UNCONFIRMED)

    def test_initials_record_verifies_with_diacritic_fold(self, default_checker):
        # "Alenlöv" folds to the record's surname key; initials confirm.
        record = self._rec(self.SHAPES["initials"])
        comps = default_checker._compare_all_fields(self.ENTRY, record)
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.VERIFIED


class TestTruePositivesPreserved:
    """The benchmark's swapped_authors (95-98% DR) and given_name_substitution
    true positives must keep firing after the alphabetization gates."""

    def test_given_name_substitution_still_routes_end_to_end(self, default_checker):
        entry = {
            "title": "Federated Learning Based on Dynamic Regularization",
            "author": "Durmus Acar and Yujing Zhao and Rafael Navarro and Matthew Mattina",
            "year": "2021",
        }
        record = _record(
            [("Durmus", "Acar"), ("Yue", "Zhao"), ("Ramon", "Navarro"), ("Matthew", "Mattina")],
            title=entry["title"],
            year=2021,
        )
        comps = default_checker._compare_all_fields(entry, record)
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.GIVEN_NAME_SUBSTITUTION

    def test_interior_swap_against_publication_order_record_still_flags(self, default_checker):
        # Same multiset, scrambled interior, record NOT alphabetized -> the
        # order rule keeps flagging (swapped_authors signature).
        entry = {
            "title": "Online Fast Adaptation and Knowledge Accumulation",
            "author": "Caccia, Massimo and Lin, Min and Rodriguez, Pau and Ostapenko, Oleksiy",
            "year": "2020",
        }
        record = _record(
            [
                ("Massimo", "Caccia"),
                ("Pau", "Rodriguez"),
                ("Oleksiy", "Ostapenko"),
                ("Min", "Lin"),
            ],
            title=entry["title"],
            year=2020,
        )
        comps = default_checker._compare_all_fields(entry, record)
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.AUTHOR_MISMATCH

    def test_same_surname_swap_against_publication_order_record_still_flags(self, default_checker):
        # SDEdit shape end-to-end: the two Songs swapped, record in
        # publication order -> still AUTHOR_MISMATCH.
        entry = {
            "title": "SDEdit: Guided Image Synthesis",
            "author": "Chenlin Meng and Yutong He and Jiaming Song and Yang Song and Jun-Yan Zhu",
            "year": "2022",
        }
        record = _record(
            [
                ("Chenlin", "Meng"),
                ("Yutong", "He"),
                ("Yang", "Song"),
                ("Jiaming", "Song"),
                ("Jun-Yan", "Zhu"),
            ],
            title=entry["title"],
            year=2022,
        )
        comps = default_checker._compare_all_fields(entry, record)
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.AUTHOR_MISMATCH
