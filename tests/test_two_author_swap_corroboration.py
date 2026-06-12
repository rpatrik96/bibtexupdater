"""2-author same-multiset order swaps need cross-source corroboration.

HALLMARK FP (verified): the valid NeurIPS 2023 entry "Rewiring Neurons in
Non-Stationary Environments" by "Zhicheng Sun and Yadong Mu" was flagged
author_mismatch. Crossref's NeurIPS proceedings deposits (10.52202 prefix)
alphabetize their contributor lists; ``_looks_alphabetized`` requires >= 3
names, so a 2-author record sorted [mu, sun] vs the entry's [sun, mu] hit the
same-multiset-different-order branch of ``symmetric_author_match`` -- a
MISMATCH minted from ONE source whose ordering is a sort artifact. With two
authors, alphabetical order coincides with publication order half the time, so
a single record fundamentally cannot decide swap vs artifact.

Default mode now keeps the MISMATCH only when a SECOND order-reliable source
independently shows the same non-entry order; otherwise the comparison softens
to PARTIAL -> UNCONFIRMED (abstention -- never VERIFIED, never a single-source
positive flag). The matcher itself stays pure (no cross-source view); the gate
lives at the ``_compare_all_fields`` call site. Strict mode is unchanged.
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
    return FactChecker(crossref, dblp, s2, cfg, logging.getLogger("two-author-swap-test"))


@pytest.fixture
def default_checker() -> FactChecker:
    return _make_checker(strict=False)


TITLE = "Rewiring Neurons in Non-Stationary Environments"

ENTRY = {
    "title": TITLE,
    "author": "Zhicheng Sun and Yadong Mu",
    "booktitle": "Advances in Neural Information Processing Systems",
    "year": "2023",
}

#: Crossref 10.52202 deposit shape: contributors alphabetized -> [Mu, Sun],
#: the reverse of the entry's publication order [Sun, Mu].
ALPHABETIZED = [{"given": "Yadong", "family": "Mu"}, {"given": "Zhicheng", "family": "Sun"}]
PUBLICATION_ORDER = [{"given": "Zhicheng", "family": "Sun"}, {"given": "Yadong", "family": "Mu"}]


def _record(authors, order_reliable=True, structured=True):
    return PublishedRecord(
        doi="10.52202/079017-0001",
        title=TITLE,
        authors=authors,
        journal="Advances in Neural Information Processing Systems",
        year=2023,
        order_reliable=order_reliable,
        structured_names=structured,
    )


class TestTwoAuthorSwapSoftening:
    def test_rewiring_neurons_single_source_abstains(self, default_checker):
        # The verified HALLMARK FP shape: a lone alphabetized Crossref record.
        best = _record(ALPHABETIZED)
        per_source = {"crossref": best}
        comps = default_checker._compare_all_fields(ENTRY, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        assert "not corroborated" in (comps["author"].note or "")
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.UNCONFIRMED

    def test_second_source_showing_entry_order_abstains(self, default_checker):
        # A second order-reliable source agrees with the ENTRY's order -> the
        # best record's ordering is the artifact -> abstain, no flag.
        best = _record(ALPHABETIZED)
        per_source = {"crossref": best, "openalex": _record(PUBLICATION_ORDER)}
        comps = default_checker._compare_all_fields(ENTRY, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        status = default_checker._determine_status(0.95, comps, ["crossref", "openalex"])
        assert status is FactCheckStatus.UNCONFIRMED

    def test_no_per_source_records_abstains(self, default_checker):
        # No cross-source view at all -> cannot corroborate -> abstain.
        best = _record(ALPHABETIZED)
        comps = default_checker._compare_all_fields(ENTRY, best)
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.UNCONFIRMED

    def test_order_unreliable_second_source_does_not_corroborate(self, default_checker):
        # An S2-style flat record showing the same order carries no order
        # signal -> still not corroborated -> abstain.
        best = _record(ALPHABETIZED)
        per_source = {
            "crossref": best,
            "s2": _record(ALPHABETIZED, order_reliable=False, structured=False),
        }
        comps = default_checker._compare_all_fields(ENTRY, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.PARTIAL
        status = default_checker._determine_status(0.95, comps, ["crossref", "s2"])
        assert status is FactCheckStatus.UNCONFIRMED


class TestTwoAuthorSwapCorroboratedStaysFlagged:
    def test_genuine_swap_corroborated_by_second_source(self, default_checker):
        # Two independent order-reliable sources both show [Mu, Sun] against the
        # entry's [Sun, Mu] -> a real swapped-authors defect, MISMATCH stands.
        best = _record(ALPHABETIZED)
        per_source = {
            "crossref": best,
            "dblp": _record(ALPHABETIZED, structured=False),
        }
        comps = default_checker._compare_all_fields(ENTRY, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.MISMATCH
        status = default_checker._determine_status(0.95, comps, ["crossref", "dblp"])
        assert status is FactCheckStatus.AUTHOR_MISMATCH

    def test_three_author_same_multiset_swap_unaffected(self, default_checker):
        # The gate is scoped to EXACTLY two authors: a 3-author same-multiset
        # reordering (the benchmark's swapped_authors signature, 95-98% DR)
        # still flags from a single non-alphabetized order-reliable source.
        entry = {
            "title": "Some Multi Author Paper",
            "author": "Caccia, Massimo and Lin, Min and Rodriguez, Pau",
            "year": "2020",
        }
        record = PublishedRecord(
            doi="10.1/x",
            title="Some Multi Author Paper",
            authors=[
                {"given": "Massimo", "family": "Caccia"},
                {"given": "Pau", "family": "Rodriguez"},
                {"given": "Min", "family": "Lin"},
            ],
            year=2020,
            order_reliable=True,
            structured_names=True,
        )
        per_source = {"crossref": record}
        comps = default_checker._compare_all_fields(entry, record, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.MISMATCH
        status = default_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.AUTHOR_MISMATCH

    def test_two_author_different_multiset_unaffected(self, default_checker):
        # A genuinely wrong second author (different multiset) is not an order
        # question at all -- the hard mismatch must survive a single source.
        entry = {"title": TITLE, "author": "Zhicheng Sun and Phantom Person", "year": "2023"}
        best = _record(PUBLICATION_ORDER)
        per_source = {"crossref": best}
        comps = default_checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comps["author"].outcome is MatchOutcome.MISMATCH


class TestTwoAuthorSwapStrictUnchanged:
    def test_strict_keeps_single_source_mismatch(self):
        strict_checker = _make_checker(strict=True)
        best = _record(ALPHABETIZED)
        per_source = {"crossref": best}
        comps = strict_checker._compare_all_fields(ENTRY, best, per_source_records=per_source)
        # Strict mode's asymmetric-cost policy keeps the single-source flag.
        assert comps["author"].outcome is MatchOutcome.MISMATCH
        status = strict_checker._determine_status(0.95, comps, ["crossref"])
        assert status is FactCheckStatus.AUTHOR_MISMATCH
