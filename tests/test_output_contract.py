"""Tests for the output contract: ``coverage_incomplete`` + ``p_valid``.

Two derived fields on :class:`FactCheckResult` (recomputed in
``__post_init__`` at every construction site) close two downstream-consumer
ambiguities:

* ``coverage_incomplete`` -- a NOT_FOUND/UNCONFIRMED produced while sources
  were erroring/throttled/circuit-broken used to be indistinguishable from a
  clean exhaustive miss, so throttled lookups were read as hallucinations.
* ``p_valid`` -- P(entry as cited is a real publication with correct
  metadata), the value consumers should threshold/rank on.
  ``overall_confidence`` keeps meaning "confidence the assigned status is the
  right call", which inverts direction across statuses.

All network access is faked at the client level (no live calls).
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from bibtex_updater.calibration import (
    P_VALID_ABSTAIN_STATUSES,
    P_VALID_NEUTRAL,
    P_VALID_NOT_FOUND_CLEAN,
    P_VALID_PROBLEM_STATUSES,
    P_VALID_VALID_STATUSES,
    p_valid_from_result,
)
from bibtex_updater.fact_checker import (
    ABSTAINED_STATUS_VALUES,
    FactChecker,
    FactCheckerConfig,
    FactCheckProcessor,
    FactCheckResult,
    FactCheckStatus,
    _compute_coverage_incomplete,
    build_verification_result,
)

LOGGER = logging.getLogger("test_output_contract")


def _result(
    status: FactCheckStatus,
    *,
    errors: list[str] | None = None,
    confidence: float = 0.0,
    key: str = "k",
) -> FactCheckResult:
    return FactCheckResult(
        entry_key=key,
        entry_type="article",
        status=status,
        overall_confidence=confidence,
        field_comparisons={},
        best_match=None,
        api_sources_queried=["crossref", "dblp"],
        api_sources_with_hits=[],
        errors=list(errors or []),
    )


# ===========================================================================
# Task 1: coverage_incomplete derivation
# ===========================================================================


class TestCoverageIncomplete:
    def test_not_found_with_source_error_is_incomplete(self):
        res = _result(FactCheckStatus.NOT_FOUND, errors=["DBLP: timeout"])
        assert res.coverage_incomplete is True

    def test_clean_not_found_is_complete(self):
        res = _result(FactCheckStatus.NOT_FOUND)
        assert res.coverage_incomplete is False

    def test_unconfirmed_with_openalex_error_is_incomplete(self):
        res = _result(FactCheckStatus.UNCONFIRMED, errors=["OpenAlex: 429 Too Many Requests"])
        assert res.coverage_incomplete is True

    def test_verified_with_source_error_stays_false(self):
        """A positive verdict stands on its evidence even when a source errored."""
        res = _result(FactCheckStatus.VERIFIED, errors=["DBLP: timeout"], confidence=0.9)
        assert res.coverage_incomplete is False

    def test_problem_status_with_source_error_stays_false(self):
        for status in (
            FactCheckStatus.HALLUCINATED,
            FactCheckStatus.AUTHOR_MISMATCH,
            FactCheckStatus.DOI_MISMATCH,
        ):
            res = _result(status, errors=["Crossref: boom"], confidence=0.8)
            assert res.coverage_incomplete is False, status

    def test_api_error_always_incomplete(self):
        """API_ERROR is definitionally incomplete coverage, errors or not."""
        assert _result(FactCheckStatus.API_ERROR).coverage_incomplete is True
        assert _result(FactCheckStatus.API_ERROR, errors=["Exception: x"]).coverage_incomplete is True

    def test_circuit_open_error_text_counts(self):
        """The per-entry errors list carries CircuitOpenError texts from
        _query_cascade; any entry there marks the abstention incomplete."""
        res = _result(
            FactCheckStatus.NOT_FOUND,
            errors=["Crossref: CircuitOpenError: crossref circuit open for 30s"],
        )
        assert res.coverage_incomplete is True

    def test_strict_warn_cnv_promotion_keeps_the_signal(self):
        """--strict-warn-cnv re-labels NOT_FOUND/UNCONFIRMED; the throttling
        signal must survive the promotion (it matters MOST when CI fails on
        these entries)."""
        assert _result(FactCheckStatus.STRICT_WARN_CNV, errors=["DBLP: 429"]).coverage_incomplete is True
        assert _result(FactCheckStatus.STRICT_WARN_CNV).coverage_incomplete is False

    def test_every_abstained_status_with_errors_is_incomplete(self):
        for value in ABSTAINED_STATUS_VALUES:
            res = _result(FactCheckStatus(value), errors=["OpenAlex: 503"])
            assert res.coverage_incomplete is True, value

    def test_helper_and_field_agree_for_every_status(self):
        """__post_init__ must apply exactly the module-level rule."""
        for status in FactCheckStatus:
            for errors in ([], ["DBLP: timeout"]):
                expected = _compute_coverage_incomplete(status, errors)
                assert _result(status, errors=errors).coverage_incomplete is expected, (status, errors)

    def test_constructor_values_are_normalized_to_the_derivation(self):
        """The fields are DERIVED: explicitly passed values that contradict the
        verdict are normalized away, so inconsistent results cannot exist."""
        res = FactCheckResult(
            entry_key="k",
            entry_type="article",
            status=FactCheckStatus.VERIFIED,
            overall_confidence=0.9,
            field_comparisons={},
            best_match=None,
            api_sources_queried=[],
            api_sources_with_hits=[],
            errors=["DBLP: timeout"],
            coverage_incomplete=True,  # contradicts VERIFIED -> normalized
            p_valid=0.0,  # contradicts the contract -> recomputed
        )
        assert res.coverage_incomplete is False
        assert res.p_valid == pytest.approx(0.95)


# ===========================================================================
# Task 2: p_valid wiring on the result object
# ===========================================================================


class TestPValidOnResult:
    def test_matches_function_for_every_status_and_confidence(self):
        for status in FactCheckStatus:
            for conf in (0.0, 0.45, 0.78, 0.93, 1.0):
                for errors in ([], ["OpenAlex: 429"]):
                    res = _result(status, errors=errors, confidence=conf)
                    expected = p_valid_from_result(status.value, conf, res.coverage_incomplete)
                    assert res.p_valid == pytest.approx(expected), (status, conf, errors)

    def test_verified_high_confidence_maps_high(self):
        assert _result(FactCheckStatus.VERIFIED, confidence=0.88).p_valid == pytest.approx(0.94)

    def test_problem_high_confidence_maps_low(self):
        assert _result(FactCheckStatus.HALLUCINATED, confidence=0.93).p_valid == pytest.approx(0.035)

    def test_not_found_clean_vs_coverage_incomplete(self):
        clean = _result(FactCheckStatus.NOT_FOUND, confidence=0.45)
        throttled = _result(FactCheckStatus.NOT_FOUND, errors=["DBLP: 429"], confidence=0.45)
        assert clean.p_valid == pytest.approx(P_VALID_NOT_FOUND_CLEAN)
        assert throttled.p_valid == pytest.approx(P_VALID_NEUTRAL)

    def test_build_verification_result_carries_the_contract(self):
        res = _result(FactCheckStatus.NOT_FOUND, errors=["DBLP: timeout"], confidence=0.45)
        rich = build_verification_result(res)
        assert rich.p_valid == pytest.approx(res.p_valid)
        assert rich.coverage_incomplete is True


# ===========================================================================
# Polarity completeness: future statuses cannot silently default
# ===========================================================================


class TestPolarityCompleteness:
    def test_every_status_in_exactly_one_polarity_or_special_case(self):
        """Every FactCheckStatus value must be claimed by exactly one polarity
        set, or be the explicit ``not_found`` special case. A new status that
        is added without a polarity decision fails here instead of silently
        falling into the defensive-neutral branch."""
        special = {FactCheckStatus.NOT_FOUND.value}
        for status in FactCheckStatus:
            memberships = [
                status.value in P_VALID_VALID_STATUSES,
                status.value in P_VALID_PROBLEM_STATUSES,
                status.value in P_VALID_ABSTAIN_STATUSES,
                status.value in special,
            ]
            assert sum(memberships) == 1, f"{status.value}: claimed by {sum(memberships)} polarity sets"

    def test_polarity_sets_are_disjoint(self):
        assert not (P_VALID_VALID_STATUSES & P_VALID_PROBLEM_STATUSES)
        assert not (P_VALID_VALID_STATUSES & P_VALID_ABSTAIN_STATUSES)
        assert not (P_VALID_PROBLEM_STATUSES & P_VALID_ABSTAIN_STATUSES)

    def test_polarity_sets_contain_no_unknown_statuses(self):
        known = {s.value for s in FactCheckStatus}
        for name, polarity in (
            ("valid", P_VALID_VALID_STATUSES),
            ("problem", P_VALID_PROBLEM_STATUSES),
            ("abstain", P_VALID_ABSTAIN_STATUSES),
        ):
            unknown = set(polarity) - known
            assert not unknown, f"{name} polarity names unknown statuses: {unknown}"

    def test_abstained_statuses_are_abstention_polarity(self):
        """Every JSONL ``abstained`` status must price at neutral (or the
        not_found special case) -- an abstention can never be valid/problem
        polarity."""
        for value in ABSTAINED_STATUS_VALUES:
            assert value in P_VALID_ABSTAIN_STATUSES or value == FactCheckStatus.NOT_FOUND.value, value


# ===========================================================================
# End-to-end through check_entry (fakes only)
# ===========================================================================

ENTRY_TITLE = "Robust Widget Learning at Scale"


def _entry() -> dict[str, str]:
    return {
        "ID": "widget2023",
        "ENTRYTYPE": "article",
        "title": ENTRY_TITLE,
        "author": "Smith, Jane and Doe, John",
        "journal": "Journal of Widget Research",
        "year": "2023",
    }


def _junk_crossref_message() -> dict:
    """A retrievable but unrelated record: keeps the candidate pool non-empty
    so the verdict is the scored NOT_FOUND abstention, not API_ERROR."""
    return {
        "DOI": "10.9999/unrelated.2019",
        "type": "journal-article",
        "title": ["Comparative Taxonomy of Deep Sea Fish"],
        "author": [{"given": "Zoe", "family": "Quux"}],
        "container-title": ["Journal of Marine Biology"],
        "issued": {"date-parts": [[2019]]},
    }


def _build_checker(*, cr_search, oa_search) -> FactChecker:
    """FactChecker with MagicMock clients; cr_search/oa_search are either a
    list of items to return or an Exception instance to raise."""
    crossref = MagicMock()
    if isinstance(cr_search, Exception):
        crossref.search.side_effect = cr_search
    else:
        crossref.search.return_value = list(cr_search)
    crossref.http = MagicMock()
    crossref.http.s2_api_key = None  # keep the key-gated S2 match step off
    dblp = MagicMock()
    dblp.search.return_value = []
    s2 = MagicMock()
    s2.search.return_value = []
    s2.get_paper.return_value = None
    openalex = MagicMock()
    if isinstance(oa_search, Exception):
        openalex.search.side_effect = oa_search
    else:
        openalex.search.return_value = list(oa_search)
    openalex.search_sources.return_value = []
    openreview = MagicMock()
    openreview.search.return_value = []
    return FactChecker(
        crossref,
        dblp,
        s2,
        FactCheckerConfig(),
        LOGGER,
        openalex=openalex,
        openreview=openreview,
    )


class TestCheckEntryEndToEnd:
    def test_not_found_with_throttled_source_is_coverage_incomplete(self):
        """Crossref retrieves only an unrelated low-score candidate while
        OpenAlex is throttled: the NOT_FOUND abstention must carry
        coverage_incomplete=True and a neutral p_valid."""
        checker = _build_checker(
            cr_search=[_junk_crossref_message()],
            oa_search=Exception("429 Too Many Requests"),
        )
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.NOT_FOUND
        assert any("OpenAlex" in e for e in result.errors)
        assert result.coverage_incomplete is True
        assert result.p_valid == pytest.approx(P_VALID_NEUTRAL)

    def test_clean_exhaustive_miss_is_complete_coverage(self):
        """Every source answered and none knows the paper: a clean exhaustive
        NOT_FOUND with the below-neutral p_valid."""
        checker = _build_checker(cr_search=[], oa_search=[])
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.NOT_FOUND
        assert result.errors == []
        assert result.coverage_incomplete is False
        assert result.p_valid == pytest.approx(P_VALID_NOT_FOUND_CLEAN)

    def test_all_sources_erroring_is_api_error_and_incomplete(self):
        checker = _build_checker(
            cr_search=Exception("503 Service Unavailable"),
            oa_search=Exception("503 Service Unavailable"),
        )
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.API_ERROR
        assert result.coverage_incomplete is True
        assert result.p_valid == pytest.approx(P_VALID_NEUTRAL)


# ===========================================================================
# Serialization: JSONL (streamed + batch), JSON report, summary
# ===========================================================================


class _FakeChecker:
    """Duck-typed checker returning prepared results by entry ID."""

    def __init__(self, results: dict[str, FactCheckResult]):
        self._results = results

    def check_entry(self, entry):
        return self._results[entry["ID"]]


def _contract_results() -> tuple[list[dict[str, str]], list[FactCheckResult]]:
    entries = [
        {"ID": "throttled", "ENTRYTYPE": "article", "title": "A"},
        {"ID": "clean", "ENTRYTYPE": "article", "title": "B"},
        {"ID": "good", "ENTRYTYPE": "article", "title": "C"},
        {"ID": "broken", "ENTRYTYPE": "article", "title": "D"},
    ]
    results = [
        _result(FactCheckStatus.NOT_FOUND, errors=["DBLP: 429"], confidence=0.45, key="throttled"),
        _result(FactCheckStatus.NOT_FOUND, confidence=0.45, key="clean"),
        _result(FactCheckStatus.VERIFIED, confidence=0.88, key="good"),
        _result(FactCheckStatus.API_ERROR, errors=["Exception: boom"], key="broken"),
    ]
    return entries, results


class TestSerialization:
    def test_streamed_jsonl_carries_the_contract_keys(self, tmp_path):
        entries, results = _contract_results()
        checker = _FakeChecker({r.entry_key: r for r in results})
        processor = FactCheckProcessor(checker, LOGGER)
        jsonl_path = tmp_path / "out.jsonl"
        processor.process_entries(entries, jsonl_path=str(jsonl_path))

        lines = {rec["key"]: rec for rec in (json.loads(line) for line in jsonl_path.read_text().splitlines() if line)}
        assert lines["throttled"]["coverage_incomplete"] is True
        assert lines["throttled"]["abstained"] is True
        assert lines["throttled"]["p_valid"] == pytest.approx(P_VALID_NEUTRAL)
        assert lines["clean"]["coverage_incomplete"] is False
        assert lines["clean"]["p_valid"] == pytest.approx(P_VALID_NOT_FOUND_CLEAN)
        assert lines["good"]["coverage_incomplete"] is False
        assert lines["good"]["p_valid"] == pytest.approx(0.94)
        assert lines["broken"]["coverage_incomplete"] is True

    def test_generate_jsonl_matches_results(self):
        _entries, results = _contract_results()
        processor = FactCheckProcessor(MagicMock(), LOGGER)
        records = [json.loads(line) for line in processor.generate_jsonl(results)]
        for rec, res in zip(records, results):
            assert rec["key"] == res.entry_key
            assert rec["coverage_incomplete"] is res.coverage_incomplete
            assert rec["p_valid"] == pytest.approx(res.p_valid)
            # The emitted p_valid must be reconstructible from the same line.
            assert rec["p_valid"] == pytest.approx(
                p_valid_from_result(rec["status"], rec["confidence"], rec["coverage_incomplete"])
            )

    def test_generate_json_report_per_entry_dicts(self):
        _entries, results = _contract_results()
        processor = FactCheckProcessor(MagicMock(), LOGGER)
        report = processor.generate_json_report(results)
        by_key = {e["key"]: e for e in report["entries"]}
        assert by_key["throttled"]["coverage_incomplete"] is True
        assert by_key["throttled"]["p_valid"] == pytest.approx(P_VALID_NEUTRAL)
        assert by_key["good"]["coverage_incomplete"] is False
        assert by_key["good"]["p_valid"] == pytest.approx(0.94)
        assert report["summary"]["coverage_incomplete_count"] == 2

    def test_generate_summary_counts_incomplete_coverage(self):
        _entries, results = _contract_results()
        processor = FactCheckProcessor(MagicMock(), LOGGER)
        summary = processor.generate_summary(results)
        # throttled NOT_FOUND + API_ERROR; the clean NOT_FOUND and the
        # VERIFIED-with-no-errors do not count.
        assert summary["coverage_incomplete_count"] == 2

    def test_generate_summary_zero_when_all_clean(self):
        processor = FactCheckProcessor(MagicMock(), LOGGER)
        results = [
            _result(FactCheckStatus.VERIFIED, confidence=0.9),
            _result(FactCheckStatus.NOT_FOUND),
        ]
        assert processor.generate_summary(results)["coverage_incomplete_count"] == 0
