"""A paper renamed after submission does not make its citation wrong.

arXiv titles change. A preprint is cited, the authors retitle it for the
conference version, and the entry now disagrees with the current arXiv record
while being a faithful record of what the work was called when it was cited.
``_check_arxiv_id_consistency`` saw only the current title and reported
ARXIV_ID_MISMATCH.

Measured on the 2026-09 InterpScience screening: six agents hand-adjudicated the
300 flagged references carrying only automated evidence, and 178 were correct as
cited with preprint-to-publication retitling the single dominant cause.

The negatives matter as much as the positives. Wrong titles were 14 of the 81
real errors in that same set, so a fix that made title matching permissive would
cost more than the bug. Both negative fixtures here are real: entries whose
cited title no version ever carried.
"""

from __future__ import annotations

import pytest

from bibtex_updater.fact_checker import ArxivClient, FactChecker, FactCheckerConfig
from bibtex_updater.utils import SourceUnavailableError


def abs_page(title: str, versions: int, arxiv_id: str = "2308.10248") -> str:
    """An arXiv abstract page, trimmed to the two parts we read."""
    history = "".join(
        f'<strong><a href="/abs/{arxiv_id}v{n}" rel="nofollow">[v{n}]</a></strong>'
        f" Sun, 20 Aug 2023 12:2{n}:05 UTC (395 KB)<br/>"
        for n in range(1, versions + 1)
    )
    return (
        f'<html><head><meta name="citation_title" content="{title}"/></head>'
        f'<body><div class="submission-history"><h2>Submission history</h2>{history}</div></body></html>'
    )


# 2308.10248: v1 "Activation Addition: ..." -> current "Steering Language Models With ..."
ACTIVATION_ADDITION_V1 = "Activation Addition: Steering Language Models Without Optimization"
ACTIVATION_ADDITION_NOW = "Steering Language Models With Activation Engineering"

# 2304.14767: every version carries the same title; the cited "... in GPT-2" is not one of them.
DISSECTING_ALL = "Dissecting Recall of Factual Associations in Auto-Regressive Language Models"
DISSECTING_CITED = "Dissecting recall of factual associations in GPT-2"

# 2106.02997: "Causal Abstractions of Neural Networks" in v1 and current alike.
CAUSAL_ABSTRACTIONS_ALL = "Causal Abstractions of Neural Networks"
CAUSAL_ABSTRACTIONS_CITED = "Causal abstractions of neural networks with interchange interventions"


class FakeHttp:
    """Serves abs pages from a {version: title} map and counts every fetch."""

    def __init__(self, per_version: dict[int, str], versions: int, arxiv_id: str = "2308.10248"):
        self.per_version = per_version
        self.versions = versions
        self.arxiv_id = arxiv_id
        self.fetched: list[str] = []

    def _request(self, method, url, **kwargs):
        self.fetched.append(url)
        version = int(url.rsplit("v", 1)[-1])
        title = self.per_version.get(version)

        class R:
            status_code = 200 if title is not None else 404
            text = abs_page(title, self.versions, self.arxiv_id) if title is not None else ""

        return R()


def make_checker(http, **cfg) -> FactChecker:
    checker = FactChecker.__new__(FactChecker)
    checker.config = FactCheckerConfig(**cfg)
    checker.arxiv = ArxivClient(http)
    import logging

    checker.logger = logging.getLogger("test")
    return checker


class TestVersionTitleFetch:
    def test_reads_the_title_of_a_named_version(self):
        http = FakeHttp({1: ACTIVATION_ADDITION_V1}, versions=5)
        assert ArxivClient(http).fetch_version_title("2308.10248", 1) == ACTIVATION_ADDITION_V1

    def test_reads_the_version_count_from_the_same_page(self):
        http = FakeHttp({1: ACTIVATION_ADDITION_V1}, versions=5)
        _, count = ArxivClient(http).fetch_version_title_and_count("2308.10248", 1)
        assert count == 5

    def test_a_missing_version_is_none_not_an_error(self):
        http = FakeHttp({1: ACTIVATION_ADDITION_V1}, versions=1)
        assert ArxivClient(http).fetch_version_title("2308.10248", 9) is None


class TestRetitlingIsNotAMismatch:
    def test_cited_title_matching_v1_clears_the_finding(self):
        http = FakeHttp({1: ACTIVATION_ADDITION_V1}, versions=5)
        checker = make_checker(http)
        matched = checker._cited_title_matches_prior_version("2308.10248", ACTIVATION_ADDITION_V1)
        assert matched == 1

    def test_the_common_case_costs_one_fetch(self):
        http = FakeHttp({1: ACTIVATION_ADDITION_V1}, versions=5)
        checker = make_checker(http)
        checker._cited_title_matches_prior_version("2308.10248", ACTIVATION_ADDITION_V1)
        assert len(http.fetched) == 1, "v1 carries both its title and the history"

    def test_matches_a_middle_version_too(self):
        http = FakeHttp(
            {1: "Some Early Working Title", 2: ACTIVATION_ADDITION_V1, 3: ACTIVATION_ADDITION_NOW},
            versions=3,
        )
        checker = make_checker(http)
        assert checker._cited_title_matches_prior_version("2308.10248", ACTIVATION_ADDITION_V1) == 2

    def test_minor_spelling_variation_still_matches(self):
        # The real entry spells it "Optimisation"; arXiv says "Optimization".
        http = FakeHttp({1: ACTIVATION_ADDITION_V1}, versions=5)
        checker = make_checker(http)
        cited = "Activation Addition: Steering Language Models Without Optimisation"
        assert checker._cited_title_matches_prior_version("2308.10248", cited) == 1


class TestWrongTitlesStayFlagged:
    """The 14 real wrong-title errors must survive this fix."""

    def test_a_title_no_version_ever_carried_is_not_cleared(self):
        http = FakeHttp({1: DISSECTING_ALL, 2: DISSECTING_ALL, 3: DISSECTING_ALL}, versions=3)
        checker = make_checker(http)
        assert checker._cited_title_matches_prior_version("2304.14767", DISSECTING_CITED) is None

    def test_causal_abstractions_stays_flagged(self):
        http = FakeHttp({1: CAUSAL_ABSTRACTIONS_ALL, 2: CAUSAL_ABSTRACTIONS_ALL}, versions=2)
        checker = make_checker(http)
        assert checker._cited_title_matches_prior_version("2106.02997", CAUSAL_ABSTRACTIONS_CITED) is None

    def test_a_wholly_unrelated_title_is_not_cleared(self):
        http = FakeHttp({1: ACTIVATION_ADDITION_V1}, versions=2)
        checker = make_checker(http)
        assert checker._cited_title_matches_prior_version("2308.10248", "Attention Is All You Need") is None


class TestBudget:
    def test_fetches_are_bounded(self):
        titles = {n: f"Working Title {n}" for n in range(1, 30)}
        http = FakeHttp(titles, versions=29)
        checker = make_checker(http, arxiv_max_version_fetches=4)
        checker._cited_title_matches_prior_version("2308.10248", "Nothing Like Any Of Them")
        assert len(http.fetched) <= 4

    def test_disabled_by_config_fetches_nothing(self):
        http = FakeHttp({1: ACTIVATION_ADDITION_V1}, versions=5)
        checker = make_checker(http, check_arxiv_version_history=False)
        assert checker._cited_title_matches_prior_version("2308.10248", ACTIVATION_ADDITION_V1) is None
        assert http.fetched == []

    def test_never_fetches_the_current_version(self):
        # The current title is already known from the API record; re-fetching it
        # would spend budget to learn nothing.
        http = FakeHttp({n: f"T{n}" for n in range(1, 4)}, versions=3)
        checker = make_checker(http)
        checker._cited_title_matches_prior_version("2308.10248", "Nothing")
        assert not any(url.endswith("v3") for url in http.fetched)


class TestUnavailableSourceDoesNotConvict:
    """Silence from arXiv must not read as "no version carried this title"."""

    class DeadHttp:
        def _request(self, method, url, **kwargs):
            raise OSError("arXiv unreachable")

    def test_a_failed_fetch_raises_rather_than_returning_none(self):
        with pytest.raises(SourceUnavailableError):
            ArxivClient(self.DeadHttp()).fetch_version_title("2308.10248", 1)

    def test_the_checker_abstains_instead_of_clearing(self):
        checker = make_checker(self.DeadHttp())
        # None here means the mismatch finding stands, which is the safe
        # direction: an unreachable source cannot clear a flag.
        assert checker._cited_title_matches_prior_version("2308.10248", "Anything At All") is None


class TestThresholdSeparation:
    """The version threshold must be stricter than the mismatch threshold.

    `arxiv_consistency_min_title` (0.50) asks whether two titles might be the
    same paper. Clearing a finding asserts the paper actually carried the cited
    title, which is a stronger claim. At 0.50 both real wrong-title errors below
    would be cleared, so the separation is the whole safety margin of this fix.
    """

    # (cited title, the title arXiv holds, should_clear)
    CASES = [
        (
            "Activation Addition: Steering Language Models Without Optimisation",
            "Activation Addition: Steering Language Models Without Optimization",
            True,
        ),
        (
            "RelP: Faithful and Efficient Circuit Discovery via Relevance Patching",
            "RelP: Faithful and Efficient Circuit Discovery via Relevance Patching in Language Models",
            True,
        ),
        (
            "Not All Language Model Features Are Linear",
            "Not All Language Model Features Are One-Dimensionally Linear",
            True,
        ),
        (
            "Identifiability in Exact Two-Layer Sparse Matrix Factorization",
            "Identifiability in Two-Layer Sparse Matrix Factorization",
            True,
        ),
        (DISSECTING_CITED, DISSECTING_ALL, False),
        (CAUSAL_ABSTRACTIONS_CITED, CAUSAL_ABSTRACTIONS_ALL, False),
        ("Attention Is All You Need", ACTIVATION_ADDITION_V1, False),
    ]

    @pytest.mark.parametrize("cited,held,should_clear", CASES)
    def test_measured_cases_land_on_the_right_side(self, cited, held, should_clear):
        http = FakeHttp({1: held}, versions=3)
        checker = make_checker(http)
        matched = checker._cited_title_matches_prior_version("2308.10248", cited)
        assert (matched is not None) is should_clear

    def test_the_two_thresholds_have_not_been_collapsed(self):
        cfg = FactCheckerConfig()
        assert cfg.arxiv_version_title_min > cfg.arxiv_consistency_min_title
