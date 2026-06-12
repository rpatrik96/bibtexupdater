"""Tests for the key-gated Semantic Scholar ``/paper/search/match`` cascade step.

With an S2 API key on the shared HTTP client, the cascade consults
``/paper/search/match`` (single best title match, one round-trip) immediately
AFTER the Crossref step and BEFORE OpenAlex, and skips the final S2
relevance-search step whenever the match step contributed a record. Without a
key the cascade is byte-for-byte the legacy order. A 404 from /match is a
normal "no match found" miss, never an error.

All fakes; behavior is proven by counting calls on the fakes.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from bibtex_updater.fact_checker import (
    FactChecker,
    FactCheckerConfig,
    SemanticScholarClient,
)

# ------------- Helpers -------------


def _entry(journal="JML"):
    e = {
        "ID": "x",
        "ENTRYTYPE": "article",
        "title": "Deep Learning",
        "author": "Smith, John",
        "year": "2024",
    }
    if journal:
        e["journal"] = journal
    return e


def _s2_item(venue="JML"):
    return {
        "title": "Deep Learning",
        "authors": [{"name": "John Smith"}],
        "venue": venue,
        "year": 2024,
        "externalIds": {"DOI": "10.1234/x"},
    }


def _build_checker(s2_api_key, match_items=(), search_items=(), cr_items=()):
    """FactChecker over MagicMock clients; ``s2_api_key`` lands on crossref.http."""
    crossref = MagicMock()
    crossref.search.return_value = list(cr_items)
    crossref.http = MagicMock()
    # A MagicMock attribute is truthy but NOT a str; the cascade requires a real
    # string key, so set it explicitly either way.
    crossref.http.s2_api_key = s2_api_key
    dblp = MagicMock()
    dblp.search.return_value = []
    s2 = MagicMock()
    s2.search.return_value = list(search_items)
    s2.match_title.return_value = list(match_items)
    openalex = MagicMock()
    openalex.search.return_value = []
    openreview = MagicMock()
    openreview.search.return_value = []
    fc = FactChecker(
        crossref,
        dblp,
        s2,
        FactCheckerConfig(top_k=3),
        logging.getLogger("test_s2_match"),
        openalex=openalex,
        openreview=openreview,
    )
    return fc


def _run_cascade(fc, entry=None):
    entry = entry or _entry()
    sources_queried: list[str] = []
    sources_with_hits: list[str] = []
    errors: list[str] = []
    candidates = fc._query_cascade(entry, "deep learning smith", sources_queried, sources_with_hits, errors)
    return candidates, sources_queried, sources_with_hits, errors


# ------------- With an API key -------------


class TestMatchStepWithKey:
    def test_match_step_queried_after_crossref_before_openalex(self):
        # Match returns the right paper but with a venue that cannot confirm the
        # entry's claimed venue -> no full confirmation -> cascade continues, so
        # the relative order of every step is observable.
        fc = _build_checker("real-key", match_items=[_s2_item(venue="NeurIPS")])
        _, sources_queried, _, _ = _run_cascade(fc, _entry(journal="ICML"))

        assert sources_queried[:3] == ["crossref", "semanticscholar", "openalex"]
        assert fc.s2.match_title.call_count == 1
        # The match step received the LaTeX-stripped raw title.
        assert fc.s2.match_title.call_args[0][0] == "Deep Learning"

    def test_full_confirmation_short_circuits_after_match(self):
        fc = _build_checker("real-key", match_items=[_s2_item(venue="JML")])
        candidates, sources_queried, sources_with_hits, errors = _run_cascade(fc)

        assert candidates
        assert sources_queried == ["crossref", "semanticscholar"]
        assert sources_with_hits == ["semanticscholar"]
        # Nothing after the match step was touched.
        assert fc.openalex.search.call_count == 0
        assert fc.dblp.search.call_count == 0
        assert fc.openreview.search.call_count == 0
        assert fc.s2.search.call_count == 0
        assert errors == []

    def test_final_s2_search_skipped_when_match_contributed(self):
        # Match hit (but not a full confirmation: venue differs) -> the cascade
        # walks every remaining source yet never re-queries S2 relevance search.
        fc = _build_checker("real-key", match_items=[_s2_item(venue="NeurIPS")])
        _, sources_queried, _, _ = _run_cascade(fc, _entry(journal="ICML"))

        assert fc.s2.match_title.call_count == 1
        assert fc.s2.search.call_count == 0
        # S2 is reported once, not twice.
        assert sources_queried.count("semanticscholar") == 1
        assert sources_queried[:5] == ["crossref", "semanticscholar", "openalex", "dblp", "openreview"]

    def test_final_s2_search_still_runs_when_match_missed(self):
        # /match returned nothing (e.g. a 404 "no match found") -> the final
        # relevance-search step runs exactly as before.
        fc = _build_checker("real-key", match_items=[], search_items=[])
        _, sources_queried, _, errors = _run_cascade(fc)

        assert fc.s2.match_title.call_count == 1
        assert fc.s2.search.call_count == 1
        # No duplicate source bookkeeping, and a miss is not an error.
        assert sources_queried.count("semanticscholar") == 1
        assert errors == []

    def test_match_miss_cascade_order_preserved(self):
        fc = _build_checker("real-key", match_items=[])
        _, sources_queried, _, _ = _run_cascade(fc)

        # Same steps as the legacy cascade plus the early S2 slot; the relaxed
        # fallback retries fire because nothing was found anywhere.
        assert sources_queried == [
            "crossref",
            "semanticscholar",
            "openalex",
            "dblp",
            "openreview",
            "crossref-fallback",
            "openalex-fallback",
        ]


# ------------- Without an API key -------------


class TestNoKeyUnchanged:
    def test_zero_match_calls_and_legacy_order(self):
        fc = _build_checker(None)
        _, sources_queried, _, _ = _run_cascade(fc)

        assert fc.s2.match_title.call_count == 0
        assert sources_queried == [
            "crossref",
            "openalex",
            "dblp",
            "openreview",
            "semanticscholar",
            "crossref-fallback",
            "openalex-fallback",
        ]

    def test_blank_key_is_treated_as_no_key(self):
        fc = _build_checker("   ")
        _, sources_queried, _, _ = _run_cascade(fc)
        assert fc.s2.match_title.call_count == 0
        assert sources_queried[1] == "openalex"

    def test_non_string_key_attribute_is_treated_as_no_key(self):
        # MagicMock http clients auto-create truthy attributes; only a real
        # string key may enable the step (keeps hermetic tests hermetic).
        fc = _build_checker(MagicMock())
        _, sources_queried, _, _ = _run_cascade(fc)
        assert fc.s2.match_title.call_count == 0
        assert "openalex" == sources_queried[1]


# ------------- 404 handling on the client -------------


class TestMatchTitle404:
    def test_404_returns_empty_list(self):
        http = MagicMock()
        http._request.return_value = MagicMock(status_code=404)
        client = SemanticScholarClient(http)

        assert client.match_title("Some Unknown Title") == []
        # Exactly one request; the 404 is a terminal miss, not retried here.
        assert http._request.call_count == 1
        kwargs = http._request.call_args.kwargs
        assert kwargs["service"] == "semanticscholar"
        assert kwargs["params"]["query"] == "Some Unknown Title"
        assert kwargs["params"]["fields"] == SemanticScholarClient.FIELDS
        assert http._request.call_args[0][1].endswith("/paper/search/match")

    def test_200_returns_data_list(self):
        http = MagicMock()
        http._request.return_value = MagicMock(status_code=200, json=lambda: {"data": [{"title": "X"}]})
        client = SemanticScholarClient(http)
        assert client.match_title("X") == [{"title": "X"}]

    def test_network_error_returns_empty_list(self):
        http = MagicMock()
        http._request.side_effect = RuntimeError("boom")
        client = SemanticScholarClient(http)
        assert client.match_title("X") == []

    def test_cascade_continues_without_error_on_match_miss(self):
        # End-to-end through the cascade: a real SemanticScholarClient whose
        # transport 404s on /match must leave the errors list untouched and let
        # the cascade proceed to OpenAlex.
        http = MagicMock()
        http.s2_api_key = "real-key"
        http._request.return_value = MagicMock(status_code=404, json=lambda: {})
        crossref = MagicMock()
        crossref.search.return_value = []
        crossref.http = http
        dblp = MagicMock()
        dblp.search.return_value = []
        openalex = MagicMock()
        openalex.search.return_value = []
        openreview = MagicMock()
        openreview.search.return_value = []
        fc = FactChecker(
            crossref,
            dblp,
            SemanticScholarClient(http),
            FactCheckerConfig(top_k=3),
            logging.getLogger("test_s2_match"),
            openalex=openalex,
            openreview=openreview,
        )
        sources_queried: list[str] = []
        errors: list[str] = []
        fc._query_cascade(_entry(), "deep learning smith", sources_queried, [], errors)

        assert errors == []
        assert "openalex" in sources_queried
        assert sources_queried[:2] == ["crossref", "semanticscholar"]


# ------------- _has_full_confirmation memo (per-cascade-invocation) -------------


class TestHasFullConfirmationMemo:
    def _crossref_item(self):
        # High title+author score (>= cascade_high_confidence) but the venue
        # contradicts the entry's claim -> NOT a full confirmation, so the
        # cascade keeps walking sources and re-evaluates the stop condition
        # after every step. Without the memo the same record is re-compared at
        # each step; with it, exactly once.
        return {
            "DOI": "10.1234/abc",
            "title": ["Deep Learning"],
            "author": [{"given": "John", "family": "Smith"}],
            "container-title": ["NeurIPS"],
            "issued": {"date-parts": [[2024]]},
            "type": "proceedings-article",
        }

    def test_compare_all_fields_runs_once_per_record_in_cascade(self):
        fc = _build_checker(None, cr_items=[self._crossref_item()])
        entry = _entry(journal="ICML")

        calls = {"n": 0}
        real_compare = fc._compare_all_fields

        def counting_compare(*args, **kwargs):
            calls["n"] += 1
            return real_compare(*args, **kwargs)

        fc._compare_all_fields = counting_compare  # type: ignore[method-assign]
        candidates = fc._query_cascade(entry, "deep learning smith", [], [], [])

        # The lone high-confidence record was compared exactly ONCE despite
        # the stop condition being evaluated after crossref/openalex/dblp/
        # openreview (4 checks in the keyless cascade).
        assert any(src == "crossref" for _, _, src in candidates)
        assert calls["n"] == 1

    def test_memoized_cascade_result_identical_to_unmemoized_check(self):
        fc = _build_checker(None, cr_items=[self._crossref_item()])
        entry = _entry(journal="ICML")
        candidates = fc._query_cascade(entry, "deep learning smith", [], [], [])
        # Direct (memo-less) call agrees: this record set has no full confirmation.
        assert fc._has_full_confirmation(entry, candidates) is False

    def test_memo_dict_caches_verdict_by_record_identity(self):
        fc = _build_checker(None)
        entry = _entry(journal="JML")
        from bibtex_updater.utils import PublishedRecord

        rec = PublishedRecord(
            doi="10.1/x",
            title="Deep Learning",
            authors=[{"given": "John", "family": "Smith"}],
            journal="JML",
            year=2024,
        )
        memo: dict[int, bool] = {}
        cands = [(1.0, rec, "crossref")]
        assert fc._has_full_confirmation(entry, cands, memo=memo) is True
        assert memo == {id(rec): True}

        # Second evaluation is served from the memo: no compare call at all.
        def boom(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("memo not used")

        fc._compare_all_fields = boom  # type: ignore[method-assign]
        assert fc._has_full_confirmation(entry, cands, memo=memo) is True
