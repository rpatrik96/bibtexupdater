"""Tests for the NONEXISTENT_VENUE positive check (Task 2).

A fabricated venue has no record to contradict it, so under the comparison
model it abstains as UNCONFIRMED forever (HALLMARK ``nonexistent_venue``: 13
dev / 9 test false negatives). The new check runs ONLY on the UNCONFIRMED
abstention path: when the entry claims a non-canonicalizable venue that >= 2
sources' candidate records for the SAME paper all fail to match, it probes the
DBLP venue registry and OpenAlex ``/sources``; only when BOTH answer
successfully with zero plausible name matches does the verdict escalate to the
positive-problem status NONEXISTENT_VENUE. Any lookup error, any registry hit,
a recognized (canonicalizable) claim, or insufficient source corroboration
keeps the abstention.

All network access is faked at the client level (no live calls).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from bibtex_updater.calibration import _PROB_SOFT, STATUS_BASE_CONFIDENCE
from bibtex_updater.fact_checker import (
    ABSTAINED_STATUS_VALUES,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    FactCheckProcessor,
    FactCheckResult,
    FactCheckStatus,
    _is_abstained_status,
)
from bibtex_updater.matching import MatchOutcome
from bibtex_updater.sources import OpenAlexClient

TITLE = "Robust Widget Learning at Scale"
AUTHORS_BIB = "Smith, Jane and Doe, John and Roe, Alice and Poe, Bob"
FABRICATED_VENUE = "International Journal of Advanced Neural Computing Systems"
REAL_VENUE = "Advances in Neural Information Processing Systems"


def _entry(venue: str = FABRICATED_VENUE) -> dict[str, str]:
    return {
        "ID": "widget2023",
        "ENTRYTYPE": "article",
        "title": TITLE,
        "author": AUTHORS_BIB,
        "journal": venue,
        "year": "2023",
    }


def _crossref_stub_message() -> dict:
    """The real paper as a Crossref deposit: right title/venue/year, but only
    a 1-author stub so the full-author preprint twin outranks it (the HALLMARK
    shape where the best-scoring candidate is venue-NON_COMPARABLE)."""
    return {
        "DOI": "10.1234/real.2023",
        "type": "proceedings-article",
        "title": [TITLE],
        "author": [{"given": "Jane", "family": "Smith"}],
        "container-title": [REAL_VENUE],
        "issued": {"date-parts": [[2023]]},
    }


def _openalex_stub_work() -> dict:
    return {
        "title": TITLE,
        "publication_year": 2023,
        "authorships": [{"author": {"display_name": "Jane Smith"}}],
        "primary_location": {"source": {"display_name": REAL_VENUE}},
    }


def _s2_twin() -> dict:
    """The paper's preprint-ish twin on S2: full authors, NO venue, so it wins
    candidate selection but cannot confirm or refute the claimed venue."""
    return {
        "title": TITLE,
        "venue": None,
        "year": 2023,
        "authors": [
            {"name": "Jane Smith"},
            {"name": "John Doe"},
            {"name": "Alice Roe"},
            {"name": "Bob Poe"},
        ],
    }


def _build_checker(
    *,
    cr_items: list | None = None,
    oa_items: list | None = None,
    s2_items: list | None = None,
    dblp_venue_hits: list | None = None,
    oa_source_hits: list | None = None,
    config: FactCheckerConfig | None = None,
) -> tuple[FactChecker, MagicMock, MagicMock]:
    """FactChecker wired with MagicMock clients; returns (checker, dblp, openalex)."""
    crossref = MagicMock()
    crossref.search.return_value = list(cr_items or [])
    crossref.http = MagicMock()
    dblp = MagicMock()
    dblp.search.return_value = []
    dblp.search_venues.return_value = dblp_venue_hits if dblp_venue_hits is not None else []
    s2 = MagicMock()
    s2.search.return_value = list(s2_items or [])
    s2.get_paper.return_value = None
    openalex = MagicMock()
    openalex.search.return_value = list(oa_items or [])
    openalex.search_sources.return_value = oa_source_hits if oa_source_hits is not None else []
    openreview = MagicMock()
    openreview.search.return_value = []
    checker = FactChecker(
        crossref,
        dblp,
        s2,
        config or FactCheckerConfig(),
        logging.getLogger("test_nonexistent_venue"),
        openalex=openalex,
        openreview=openreview,
    )
    return checker, dblp, openalex


def _fabricated_venue_checker(**kwargs) -> tuple[FactChecker, MagicMock, MagicMock]:
    """The canonical scenario: 2 sources return the real (NeurIPS) paper,
    the full-author venue-less twin wins selection -> UNCONFIRMED pre-hook."""
    return _build_checker(
        cr_items=[_crossref_stub_message()],
        oa_items=[_openalex_stub_work()],
        s2_items=[_s2_twin()],
        **kwargs,
    )


# ===========================================================================
# Status plumbing: enum / buckets / calibration
# ===========================================================================


class TestStatusPlumbing:
    def test_status_value_is_fixed_contract(self):
        assert FactCheckStatus.NONEXISTENT_VENUE.value == "nonexistent_venue"

    def test_not_an_abstention(self):
        assert "nonexistent_venue" not in ABSTAINED_STATUS_VALUES
        assert _is_abstained_status(FactCheckStatus.NONEXISTENT_VENUE) is False

    def test_calibration_soft_problem_tier(self):
        assert STATUS_BASE_CONFIDENCE["nonexistent_venue"] == _PROB_SOFT

    def test_generate_summary_counts_it_problematic(self):
        processor = FactCheckProcessor(MagicMock(), logging.getLogger("t"))
        result = FactCheckResult(
            entry_key="x",
            entry_type="article",
            status=FactCheckStatus.NONEXISTENT_VENUE,
            overall_confidence=0.78,
            field_comparisons={},
            best_match=None,
            api_sources_queried=["crossref"],
            api_sources_with_hits=["crossref"],
            errors=[],
        )
        summary = processor.generate_summary([result])
        assert summary["problematic_count"] == 1
        assert summary["abstained_count"] == 0
        assert summary["status_counts"]["nonexistent_venue"] == 1

    def test_cli_flag_exists(self):
        from bibtex_updater.fact_checker import build_parser

        args = build_parser().parse_args(["refs.bib", "--no-check-venue-existence"])
        assert args.no_check_venue_existence is True
        args = build_parser().parse_args(["refs.bib"])
        assert args.no_check_venue_existence is False

    def test_config_default_enabled(self):
        assert FactCheckerConfig().check_venue_existence is True


# ===========================================================================
# Registry clients
# ===========================================================================


def _resp(status_code: int, payload: dict) -> MagicMock:
    return MagicMock(status_code=status_code, json=lambda: payload)


class TestDblpSearchVenues:
    def test_returns_hits_on_200(self):
        http = MagicMock()
        payload = {"result": {"hits": {"hit": [{"info": {"venue": "ICML", "acronym": "ICML"}}]}}}
        http._request.return_value = _resp(200, payload)
        hits = DBLPClient(http).search_venues("icml")
        assert hits is not None and len(hits) == 1
        # Routed through the shared client with the dblp service tag.
        assert http._request.call_args.kwargs.get("service") == "dblp"
        assert "search/venue/api" in http._request.call_args.args[1]

    def test_single_dict_hit_wrapped(self):
        http = MagicMock()
        payload = {"result": {"hits": {"hit": {"info": {"venue": "ICML"}}}}}
        http._request.return_value = _resp(200, payload)
        hits = DBLPClient(http).search_venues("icml")
        assert isinstance(hits, list) and len(hits) == 1

    def test_zero_hits_is_empty_list_not_none(self):
        http = MagicMock()
        http._request.return_value = _resp(200, {"result": {"hits": {}}})
        assert DBLPClient(http).search_venues("nope") == []

    def test_non_200_is_none(self):
        http = MagicMock()
        http._request.return_value = _resp(503, {})
        assert DBLPClient(http).search_venues("icml") is None

    def test_exception_is_none(self):
        http = MagicMock()
        http._request.side_effect = RuntimeError("network down")
        assert DBLPClient(http).search_venues("icml") is None


class TestOpenAlexSearchSources:
    def test_returns_results_on_200(self):
        http = MagicMock()
        http._request.return_value = _resp(200, {"results": [{"display_name": "Foo Journal"}]})
        results = OpenAlexClient(http=http).search_sources("foo journal")
        assert results is not None and len(results) == 1
        assert http._request.call_args.kwargs.get("service") == "openalex"
        assert "/sources" in http._request.call_args.args[1]

    def test_zero_hits_is_empty_list_not_none(self):
        http = MagicMock()
        http._request.return_value = _resp(200, {"results": []})
        assert OpenAlexClient(http=http).search_sources("nope") == []

    def test_non_200_is_none(self):
        http = MagicMock()
        http._request.return_value = _resp(429, {})
        assert OpenAlexClient(http=http).search_sources("foo") is None

    def test_exception_is_none(self):
        http = MagicMock()
        http._request.side_effect = RuntimeError("boom")
        assert OpenAlexClient(http=http).search_sources("foo") is None

    def test_empty_query_short_circuits(self):
        http = MagicMock()
        assert OpenAlexClient(http=http).search_sources("") == []
        http._request.assert_not_called()


# ===========================================================================
# End-to-end check_entry behaviour
# ===========================================================================


class TestNonexistentVenueCheck:
    def test_fabricated_venue_flags_nonexistent(self):
        """Spec test 1: fabricated venue, 2 sources return the real paper with
        NeurIPS venue strings, both registries return no match ->
        NONEXISTENT_VENUE (a problem, not an abstention)."""
        checker, dblp, openalex = _fabricated_venue_checker()
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.NONEXISTENT_VENUE
        assert _is_abstained_status(result.status) is False
        assert dblp.search_venues.call_count == 1
        assert openalex.search_sources.call_count == 1
        venue_cmp = result.field_comparisons["venue"]
        assert venue_cmp.outcome is MatchOutcome.MISMATCH
        assert "not found in DBLP/OpenAlex venue registries" in (venue_cmp.note or "")

    def test_dblp_registry_error_keeps_unconfirmed(self):
        """Spec test 2a: DBLP venue lookup errors (None) -> could not check ->
        stay UNCONFIRMED."""
        checker, dblp, _openalex = _fabricated_venue_checker()
        dblp.search_venues.return_value = None
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.UNCONFIRMED

    def test_dblp_registry_raises_keeps_unconfirmed(self):
        """Spec test 2b: the lookup raising is swallowed as could-not-check."""
        checker, dblp, _openalex = _fabricated_venue_checker()
        dblp.search_venues.side_effect = RuntimeError("registry down")
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.UNCONFIRMED

    def test_openalex_registry_error_keeps_unconfirmed(self):
        """Spec test 2c: DBLP finds nothing but OpenAlex errors -> a
        nonexistence verdict needs BOTH registries -> stay UNCONFIRMED."""
        checker, _dblp, openalex = _fabricated_venue_checker()
        openalex.search_sources.return_value = None
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.UNCONFIRMED

    def test_venue_found_in_openalex_sources_keeps_unconfirmed(self):
        """Spec test 3: the claimed venue is a real obscure journal known to
        OpenAlex -> the venue exists -> stay UNCONFIRMED."""
        checker, _dblp, _openalex = _fabricated_venue_checker(
            oa_source_hits=[{"display_name": FABRICATED_VENUE}],
        )
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.UNCONFIRMED

    def test_venue_found_in_dblp_registry_keeps_unconfirmed(self):
        """DBLP-side variant of spec test 3 (acronym field counts too)."""
        checker, _dblp, openalex = _fabricated_venue_checker(
            dblp_venue_hits=[{"info": {"venue": FABRICATED_VENUE, "acronym": "IJANCS"}}],
        )
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.UNCONFIRMED
        # Short-circuits on the DBLP hit; OpenAlex registry never consulted.
        assert openalex.search_sources.call_count == 0

    def test_single_reporting_source_skips_registry_calls(self):
        """Spec test 4: only 1 source returned a venue-bearing candidate ->
        insufficient corroboration that the paper is real-and-known ->
        UNCONFIRMED and NO registry calls."""
        checker, dblp, openalex = _build_checker(
            cr_items=[_crossref_stub_message()],
            oa_items=[],
            s2_items=[_s2_twin()],
        )
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.UNCONFIRMED
        assert dblp.search_venues.call_count == 0
        assert openalex.search_sources.call_count == 0

    def test_disabled_by_config_skips_check(self):
        """Spec test 5: check disabled -> UNCONFIRMED, no registry calls."""
        checker, dblp, openalex = _fabricated_venue_checker(
            config=FactCheckerConfig(check_venue_existence=False),
        )
        result = checker.check_entry(_entry())
        assert result.status is FactCheckStatus.UNCONFIRMED
        assert dblp.search_venues.call_count == 0
        assert openalex.search_sources.call_count == 0

    def test_canonicalizable_claimed_venue_never_triggers(self):
        """Spec test 6: a recognized venue claim ('NeurIPS') is the consensus
        path's territory; the existence check must never fire for it."""
        checker, dblp, openalex = _build_checker(
            cr_items=[_crossref_stub_message()],
            oa_items=[_openalex_stub_work()],
            s2_items=[_s2_twin()],
        )
        result = checker.check_entry(_entry(venue="NeurIPS"))
        # The claim agrees with both stubs' venue strings; with the twin as
        # best match the venue stays unconfirmable -> UNCONFIRMED, no probes.
        assert result.status is FactCheckStatus.UNCONFIRMED
        assert dblp.search_venues.call_count == 0
        assert openalex.search_sources.call_count == 0

    def test_wrong_but_real_venue_stays_on_consensus_path(self):
        """Spec test 7: entry says ICML, sources agree on NeurIPS -> the
        cross-source consensus flags VENUE_MISMATCH; the existence check is
        never reached (status is not UNCONFIRMED) and no registry is called."""
        checker, dblp, openalex = _build_checker(
            cr_items=[_crossref_stub_message()],
            oa_items=[_openalex_stub_work()],
            s2_items=[_s2_twin()],
        )
        result = checker.check_entry(_entry(venue="ICML"))
        assert result.status is FactCheckStatus.VENUE_MISMATCH
        assert dblp.search_venues.call_count == 0
        assert openalex.search_sources.call_count == 0

    def test_preprintish_claimed_venue_never_triggers(self):
        """An arXiv-style claimed venue is handled by the preprint trichotomy,
        never by the existence check."""
        checker, dblp, _openalex = _build_checker(
            cr_items=[_crossref_stub_message()],
            oa_items=[_openalex_stub_work()],
            s2_items=[_s2_twin()],
        )
        result = checker.check_entry(_entry(venue="arXiv preprint arXiv:2301.00001"))
        assert result.status is not FactCheckStatus.NONEXISTENT_VENUE
        assert dblp.search_venues.call_count == 0

    def test_memoization_one_probe_for_repeated_venue(self):
        """The registry probe runs once per distinct normalized venue string,
        not once per entry."""
        checker, dblp, openalex = _fabricated_venue_checker()
        first = checker.check_entry(_entry())
        second = checker.check_entry({**_entry(), "ID": "widget2023b"})
        assert first.status is FactCheckStatus.NONEXISTENT_VENUE
        assert second.status is FactCheckStatus.NONEXISTENT_VENUE
        assert dblp.search_venues.call_count == 1
        assert openalex.search_sources.call_count == 1

    def test_jsonl_status_and_abstained_flag(self, tmp_path):
        """JSONL output: the status string flows through unchanged and the
        abstained flag is False for this positive-problem status."""
        import json

        checker, _dblp, _openalex = _fabricated_venue_checker()
        processor = FactCheckProcessor(checker, logging.getLogger("t"))
        jsonl_path = tmp_path / "out.jsonl"
        processor.process_entries([_entry()], jsonl_path=str(jsonl_path))
        lines = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        assert len(lines) == 1
        assert lines[0]["status"] == "nonexistent_venue"
        assert lines[0]["abstained"] is False
        assert "venue" in lines[0]["mismatched_fields"]
