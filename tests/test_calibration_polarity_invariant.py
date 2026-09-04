"""Every status's confidence tier must agree with its P(valid) polarity.

``STATUS_BASE_CONFIDENCE`` says how sure we are the assigned status is right.
``p_valid_from_result`` turns that into P(the entry as cited is genuine), and the
direction it moves depends on which polarity set the status is in:

    VALID-polarity    -> 0.5 + 0.5 * conf
    PROBLEM-polarity  -> 0.5 - 0.5 * conf

So a status in the PROBLEM set that draws its confidence from the
CLEARLY-CORRECT anchor produces a *more* extreme "invalid" the *more* certain we
were that it was correct. ``preprint_only`` did exactly that: 0.88 from the
CLEARLY-CORRECT tier, PROBLEM polarity, giving P(valid) = 0.060 -- more
confidently invalid than ``title_mismatch`` at 0.110 and nearly as extreme as a
DOI resolving to a different paper at 0.035, when a preprint cited as published
is weaker evidence than either.

These tests pin the invariant rather than the two instances that were found, so
the next status added cannot reintroduce it.
"""

from __future__ import annotations

import pytest

from bibtex_updater.calibration import (
    P_VALID_ABSTAIN_STATUSES,
    P_VALID_NEUTRAL,
    P_VALID_PROBLEM_STATUSES,
    P_VALID_VALID_STATUSES,
    STATUS_BASE_CONFIDENCE,
    p_valid_from_result,
)

#: The anchor reserved for "we could not decide". A PROBLEM or VALID status
#: drawing this is asserting something with don't-know confidence.
_ABSTAIN_ANCHOR = 0.45


def _statuses(bucket) -> list[str]:
    return sorted(s for s in bucket if s in STATUS_BASE_CONFIDENCE)


@pytest.mark.parametrize("status", _statuses(P_VALID_PROBLEM_STATUSES))
def test_problem_statuses_yield_p_valid_below_one_half(status):
    conf = STATUS_BASE_CONFIDENCE[status]
    assert (
        p_valid_from_result(status, conf) < 0.5
    ), f"{status} is PROBLEM-polarity but P(valid) is {p_valid_from_result(status, conf)}"


@pytest.mark.parametrize("status", _statuses(P_VALID_VALID_STATUSES))
def test_valid_statuses_yield_p_valid_above_one_half(status):
    conf = STATUS_BASE_CONFIDENCE[status]
    assert p_valid_from_result(status, conf) > 0.5


def test_no_problem_status_draws_the_clearly_correct_anchor():
    """The defect that produced P(valid) = 0.060 for a preprint."""
    correct_tier = STATUS_BASE_CONFIDENCE["verified"]
    offenders = [s for s in _statuses(P_VALID_PROBLEM_STATUSES) if STATUS_BASE_CONFIDENCE[s] == correct_tier]
    assert not offenders, (
        f"{offenders} carry PROBLEM polarity but draw the CLEARLY-CORRECT anchor "
        f"({correct_tier}). p_valid_from_result reads that as confidence in a "
        "problem, so the more certain the verdict the more extreme the wrong answer."
    )


def test_a_problem_status_is_never_more_extreme_than_the_strongest_evidence():
    """Ordering sanity: nothing outranks a DOI resolving to a different paper.

    ``doi_mismatch`` and ``future_date`` are self-contained -- arithmetic, or an
    identifier that resolves elsewhere. A fuzzy or coverage-dependent verdict
    must not claim more certainty than they do.
    """
    strongest = min(
        p_valid_from_result(s, STATUS_BASE_CONFIDENCE[s])
        for s in ("doi_mismatch", "future_date", "invalid_year", "arxiv_id_mismatch")
    )
    for status in _statuses(P_VALID_PROBLEM_STATUSES):
        if status in {"doi_mismatch", "future_date", "invalid_year", "arxiv_id_mismatch", "hallucinated"}:
            continue
        p = p_valid_from_result(status, STATUS_BASE_CONFIDENCE[status])
        assert p >= strongest, (
            f"{status} yields P(valid) {p:.3f}, at least as extreme as the strongest "
            f"self-contained evidence ({strongest:.3f}). It is not that certain."
        )


def test_abstentions_stay_neutral():
    for status in _statuses(P_VALID_ABSTAIN_STATUSES):
        assert p_valid_from_result(status, STATUS_BASE_CONFIDENCE[status]) == P_VALID_NEUTRAL


#: Statuses that deliberately assert at the weak-evidence value. Each draws
#: _PROB_WEAK or _CORRECT_WEAK, which share the abstention anchor's NUMBER but
#: not its meaning: the evidence is real and thin, rather than absent.
_DELIBERATELY_WEAK = {"doi_not_found", "url_accessible"}


def test_asserting_statuses_document_why_they_sit_at_the_weak_value():
    """A verdict asserting something must not silently borrow the don't-know anchor.

    ``doi_not_found`` drew _ABSTAIN while carrying PROBLEM polarity, and
    ``url_accessible`` drew it while carrying VALID polarity -- found by this
    test, not by inspection. Both keep their numbers: a non-resolving DOI and a
    reachable URL are each weak but real evidence. They now draw _PROB_WEAK and
    _CORRECT_WEAK, so the constant states the intent and a new status cannot
    reach that value by accident.
    """
    asserting = set(_statuses(P_VALID_PROBLEM_STATUSES)) | set(_statuses(P_VALID_VALID_STATUSES))
    offenders = sorted(
        s for s in asserting if STATUS_BASE_CONFIDENCE[s] == _ABSTAIN_ANCHOR and s not in _DELIBERATELY_WEAK
    )
    assert not offenders, (
        f"{offenders} assert a polarity at the abstention anchor ({_ABSTAIN_ANCHOR}) "
        "without being declared weak evidence. Either give them a real tier or add "
        "them to _DELIBERATELY_WEAK with a reason."
    )


def test_preprint_only_is_not_the_systems_most_confident_invalid():
    """Regression on the specific number that surfaced this."""
    p = p_valid_from_result("preprint_only", STATUS_BASE_CONFIDENCE["preprint_only"])
    assert p == pytest.approx(0.11, abs=1e-9)
    assert p > p_valid_from_result("doi_mismatch", STATUS_BASE_CONFIDENCE["doi_mismatch"])
