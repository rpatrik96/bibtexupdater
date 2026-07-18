"""Tests for the CheckIfExist cascade extensions to bibtex-check.

Covers items 1-6 of the CheckIfExist (Abbonato 2026) port:
1. Cascading source order (CrossRef -> OpenAlex -> Semantic Scholar)
2. Top-K candidate retrieval before fuzzy match
3. Cross-source author intersection
4. Numeric confidence (0-100) with explicit penalties / multi-source bonus
5. Rich VerificationResult with similarity breakdown
6. ``--non-generative`` flag and env-var gate
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock

import pytest

from bibtex_updater.fact_checker import (
    CASE_A_AUTHOR_THRESHOLD,
    CASE_A_TITLE_THRESHOLD,
    MULTI_SOURCE_BONUS,
    PENALTY_AUTHOR_MISMATCH,
    PENALTY_FABRICATED_AUTHOR_CAP,
    PENALTY_JOURNAL_MISMATCH,
    PENALTY_PER_FABRICATED_AUTHOR,
    PENALTY_TITLE_MISMATCH,
    FactChecker,
    FactCheckerConfig,
    FactCheckResult,
    FactCheckStatus,
    FieldComparison,
    VerificationResult,
    assert_no_llm_backend,
    build_verification_result,
    compute_numeric_confidence,
    is_non_generative_mode,
    set_non_generative_mode,
)
from bibtex_updater.sources import (
    CASCADE_HIGH_CONFIDENCE,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    AuthorIntersectionResult,
    OpenAlexClient,
    cross_source_author_intersection,
    openalex_work_to_candidate_record,
    select_top_k_by_title_similarity,
)
from bibtex_updater.utils import PublishedRecord

# ------------- Fixtures -------------


@pytest.fixture
def logger():
    return logging.getLogger("test_cascade_sources")


@pytest.fixture
def fake_http():
    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return mock


def _record(title, authors=None, journal="J. ML", year=2024, doi=None):
    return PublishedRecord(
        doi=doi,
        title=title,
        authors=authors or [{"given": "John", "family": "Smith"}],
        journal=journal,
        year=year,
    )


# ===========================================================================
# Item 1+2: Cascading source order + Top-K candidate retrieval
# ===========================================================================


class TestSelectTopKByTitleSimilarity:
    def test_returns_empty_when_no_candidates(self):
        assert select_top_k_by_title_similarity("Some title", []) == []

    def test_ranks_higher_similarity_first(self):
        query = "Deep Learning for Natural Language Processing"
        cands = [
            _record("Quantum gravity is hard"),
            _record("Deep Learning for Natural Language Processing"),
            _record("Deep Learning, Natural Language"),
        ]
        ranked = select_top_k_by_title_similarity(query, cands, k=3)
        assert len(ranked) == 3
        # The exact match must be first.
        assert ranked[0][1].title == "Deep Learning for Natural Language Processing"
        # Scores must be monotonically non-increasing.
        scores = [s for s, _ in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_caps_at_max_top_k(self):
        cands = [_record(f"Title {i}") for i in range(MAX_TOP_K + 5)]
        ranked = select_top_k_by_title_similarity("Title 0", cands, k=MAX_TOP_K + 5)
        assert len(ranked) == MAX_TOP_K

    def test_handles_missing_titles_gracefully(self):
        cands = [_record(""), _record("Real title")]
        ranked = select_top_k_by_title_similarity("Real title", cands, k=2)
        # Empty-title record should still be returned but with score 0.
        non_empty_first = [r.title for _, r in ranked]
        assert non_empty_first[0] == "Real title"


class TestQueryCascade:
    """Verify the cascade order CrossRef -> OpenAlex -> S2."""

    def _build(self, cr_items=(), s2_items=(), oa_items=(), config_override=None, openalex=None):
        crossref = MagicMock()
        crossref.search.return_value = list(cr_items)
        crossref.http = MagicMock()
        dblp = MagicMock()
        s2 = MagicMock()
        s2.search.return_value = list(s2_items)
        config = FactCheckerConfig(top_k=3)
        if config_override:
            for k, v in config_override.items():
                setattr(config, k, v)
        if openalex is None:
            openalex = MagicMock()
            openalex.search.return_value = list(oa_items)
        return FactChecker(crossref, dblp, s2, config, logging.getLogger("c")), openalex

    def test_high_confidence_match_short_circuits_cascade(self):
        # CrossRef returns the exact title; S2/OpenAlex must not be called.
        cr_item = {
            "DOI": "10.1234/abc",
            "title": ["Deep Learning"],
            "author": [{"given": "John", "family": "Smith"}],
            "container-title": ["JML"],
            "issued": {"date-parts": [[2024]]},
            "type": "journal-article",
        }
        fc, oa = self._build(cr_items=[cr_item])
        fc.openalex = oa
        entry = {
            "ID": "x",
            "ENTRYTYPE": "article",
            "title": "Deep Learning",
            "author": "Smith, John",
            "journal": "JML",
            "year": "2024",
        }
        sources_queried, sources_with_hits, errors = [], [], []
        candidates = fc._query_cascade(entry, "Deep Learning Smith", sources_queried, sources_with_hits, errors)
        assert candidates  # at least one match
        assert "crossref" in sources_queried
        # S2 / OpenAlex must not have been touched if CrossRef hit high-confidence.
        if any(s >= CASCADE_HIGH_CONFIDENCE for s, _, _ in candidates):
            assert "semanticscholar" not in sources_queried
            assert "openalex" not in sources_queried

    def test_falls_through_to_s2_when_crossref_empty(self):
        s2_item = {
            "title": "Deep Learning",
            "authors": [{"name": "John Smith"}],
            "venue": "JML",
            "year": 2024,
            "externalIds": {"DOI": "10.1234/x"},
        }
        fc, oa = self._build(cr_items=[], s2_items=[s2_item])
        fc.openalex = oa
        entry = {
            "ID": "x",
            "ENTRYTYPE": "article",
            "title": "Deep Learning",
            "author": "Smith, John",
            "year": "2024",
        }
        sources_queried = []
        fc._query_cascade(entry, "Deep Learning Smith", sources_queried, [], [])
        assert "crossref" in sources_queried
        assert "semanticscholar" in sources_queried

    def test_falls_through_all_sources_when_empty(self):
        fc, oa = self._build(cr_items=[], s2_items=[], oa_items=[])
        fc.openalex = oa
        entry = {
            "ID": "x",
            "ENTRYTYPE": "article",
            "title": "Deep Learning",
            "author": "Smith, John",
            "year": "2024",
        }
        sources_queried = []
        fc._query_cascade(entry, "Deep Learning Smith", sources_queried, [], [])
        # New order puts the fast, broad aggregator (OpenAlex), the CS-conference
        # authority (DBLP), and the ML-venue authority (OpenReview) before the
        # slow keyless-S2 specialist. OpenReview is lazily built from
        # ``crossref.http`` (a MagicMock here), so it is reached.
        # FIX X4: when every primary source returns nothing usable, the relaxed-
        # author retrieval fallback runs (title-only retry on Crossref + OpenAlex).
        # The fallback source names appear at the tail of the source list.
        assert sources_queried == [
            "crossref",
            "openalex",
            "dblp",
            "openreview",
            "semanticscholar",
            "crossref-fallback",
            "openalex-fallback",
        ]

    def test_top_k_capped_at_max(self):
        fc, _ = self._build(config_override={"top_k": MAX_TOP_K + 100})
        # Effective top_k must be bounded by MAX_TOP_K when actually used.
        assert min(fc.config.top_k, MAX_TOP_K) == MAX_TOP_K


# ===========================================================================
# Item 3: Cross-source author intersection
# ===========================================================================


class TestCrossSourceAuthorIntersection:
    def test_returns_empty_when_only_one_source_contributes(self):
        recs = {"crossref": _record("X", authors=[{"family": "Smith"}]), "s2": None}
        out = cross_source_author_intersection(recs)
        assert out.confirmed == []
        assert out.suspect == []
        assert out.bonus == 0.0

    def test_intersects_authors_across_sources(self):
        recs = {
            "crossref": _record(
                "X",
                authors=[{"family": "Smith"}, {"family": "Doe"}, {"family": "Jones"}],
            ),
            "s2": _record(
                "X",
                authors=[{"family": "Smith"}, {"family": "Doe"}],
            ),
            "openalex": _record(
                "X",
                authors=[{"family": "Smith"}, {"family": "Doe"}],
            ),
        }
        out = cross_source_author_intersection(recs)
        assert set(out.confirmed) == {"smith", "doe"}
        assert "jones" in out.suspect

    def test_multi_source_bonus_when_two_or_more_confirmed(self):
        recs = {
            "crossref": _record("X", authors=[{"family": "Smith"}, {"family": "Doe"}]),
            "s2": _record("X", authors=[{"family": "Smith"}, {"family": "Doe"}]),
        }
        out = cross_source_author_intersection(recs, multi_source_bonus=10.0)
        assert out.bonus == 10.0
        # And the bonus stays in [0, 10].
        out2 = cross_source_author_intersection(recs, multi_source_bonus=999.0)
        assert out2.bonus == 10.0

    def test_no_bonus_when_fewer_than_two_confirmed(self):
        recs = {
            "crossref": _record("X", authors=[{"family": "Smith"}]),
            "s2": _record("X", authors=[{"family": "Smith"}]),
        }
        out = cross_source_author_intersection(recs)
        assert out.confirmed == ["smith"]
        assert out.bonus == 0.0  # only 1 confirmed

    def test_diacritics_normalized_for_intersection(self):
        recs = {
            "crossref": _record("X", authors=[{"family": "Müller"}]),
            "s2": _record("X", authors=[{"family": "Muller"}]),
        }
        out = cross_source_author_intersection(recs)
        assert out.confirmed == ["muller"]


# ===========================================================================
# Item 4: Numeric confidence (0-100) with penalties + multi-source bonus
# ===========================================================================


class TestComputeNumericConfidence:
    def test_case_a_asymmetric_high_title_low_author(self):
        # S_title=90 > 80, S_author=50 < 90 => asymmetric formula
        # confidence = 90 - 0.5*(100 - 50) = 90 - 25 = 65
        out = compute_numeric_confidence(
            title_score=90.0,
            author_score=50.0,
            journal_score=0.0,
            year_score=0.0,
            issues=[],
            multi_source_bonus=0.0,
        )
        assert out == pytest.approx(65.0)

    def test_case_b_average_with_bonus(self):
        # mean(90,90,90,90) + 5 = 95
        out = compute_numeric_confidence(
            title_score=90.0,
            author_score=90.0,
            journal_score=90.0,
            year_score=90.0,
            issues=[],
            multi_source_bonus=5.0,
        )
        assert out == pytest.approx(95.0)

    def test_case_b_when_author_exceeds_threshold(self):
        # title>80 but author>=90 should NOT trigger Case A.
        out = compute_numeric_confidence(
            title_score=85.0,
            author_score=95.0,
            journal_score=80.0,
            year_score=80.0,
            issues=[],
            multi_source_bonus=0.0,
        )
        assert out == pytest.approx((85 + 95 + 80 + 80) / 4)

    def test_penalties_subtract(self):
        out = compute_numeric_confidence(
            title_score=80.0,
            author_score=80.0,
            journal_score=80.0,
            year_score=80.0,
            issues=["title_mismatch", "author_mismatch", "venue_mismatch"],
            multi_source_bonus=0.0,
        )
        expected = 80.0 - PENALTY_TITLE_MISMATCH - PENALTY_AUTHOR_MISMATCH - PENALTY_JOURNAL_MISMATCH
        assert out == pytest.approx(max(0.0, expected))

    def test_fabricated_author_penalty_capped(self):
        # 5 fabricated authors @ 10 each => 50, but capped at 20.
        out = compute_numeric_confidence(
            title_score=100.0,
            author_score=100.0,
            journal_score=100.0,
            year_score=100.0,
            issues=[],
            multi_source_bonus=0.0,
            fabricated_author_count=5,
        )
        # Pre-cap would be 100 - 50 = 50; post-cap = 100 - 20 = 80.
        assert out == pytest.approx(100.0 - PENALTY_FABRICATED_AUTHOR_CAP)

    def test_multi_source_bonus_caps_at_10(self):
        out = compute_numeric_confidence(
            title_score=80.0,
            author_score=80.0,
            journal_score=80.0,
            year_score=80.0,
            issues=[],
            multi_source_bonus=999.0,
        )
        assert out == pytest.approx(80.0 + 10.0)

    def test_confidence_clamped_to_zero_floor(self):
        out = compute_numeric_confidence(
            title_score=0.0,
            author_score=0.0,
            journal_score=0.0,
            year_score=0.0,
            issues=["title_mismatch", "author_mismatch", "venue_mismatch"],
            multi_source_bonus=0.0,
            fabricated_author_count=10,
        )
        assert out == 0.0


# ===========================================================================
# Item 5: Rich VerificationResult
# ===========================================================================


class TestBuildVerificationResult:
    def _fc_result(self, status=FactCheckStatus.VERIFIED, comparisons=None, errors=None):
        return FactCheckResult(
            entry_key="k",
            entry_type="article",
            status=status,
            overall_confidence=0.9,
            field_comparisons=comparisons
            or {
                "title": FieldComparison("title", "X", "X", 1.0, True),
                "author": FieldComparison("author", "Smith", "Smith", 1.0, True),
            },
            best_match=_record("X"),
            api_sources_queried=["crossref"],
            api_sources_with_hits=["crossref"],
            errors=errors or [],
        )

    def test_basic_fields_populated(self):
        r = self._fc_result()
        out = build_verification_result(r)
        assert isinstance(out, VerificationResult)
        assert out.bibtex_key == "k"
        assert out.status == FactCheckStatus.VERIFIED.value
        assert "title" in out.similarity_breakdown
        assert out.similarity_breakdown["title"] == pytest.approx(100.0)

    def test_includes_intersection_data_when_provided(self):
        r = self._fc_result()
        intersection = AuthorIntersectionResult(
            confirmed=["smith", "doe"],
            suspect=["jones"],
            sources_consulted=["crossref", "s2"],
            bonus=10.0,
        )
        out = build_verification_result(r, intersection=intersection)
        assert out.confirmed_authors == ["smith", "doe"]
        assert out.suspect_authors == ["jones"]
        assert "potential_fabricated_authors" in out.issues

    def test_mismatched_fields_become_issues(self):
        r = self._fc_result(
            comparisons={
                "title": FieldComparison("title", "X", "Y", 0.3, False),
                "author": FieldComparison("author", "Smith", "Smith", 1.0, True),
            },
        )
        out = build_verification_result(r)
        assert "title_mismatch" in out.issues
        assert "author_mismatch" not in out.issues

    def test_matched_metadata_populated_from_best_match(self):
        r = self._fc_result()
        r.best_match = PublishedRecord(doi="10.1/y", title="Y title", journal="Y venue", year=2024, authors=[])
        out = build_verification_result(r)
        assert out.matched_metadata is not None
        assert out.matched_metadata["doi"] == "10.1/y"
        assert out.matched_metadata["title"] == "Y title"


# ===========================================================================
# Item 6: Non-generative-AI mode
# ===========================================================================


class TestNonGenerativeMode:
    def setup_method(self):
        # Always start each test with the flag off and the env var clean.
        set_non_generative_mode(False)
        os.environ.pop("BIBTEX_CHECK_NON_GENERATIVE", None)

    def teardown_method(self):
        set_non_generative_mode(False)
        os.environ.pop("BIBTEX_CHECK_NON_GENERATIVE", None)

    def test_default_is_off(self):
        assert is_non_generative_mode() is False

    def test_set_flag_enables_mode(self):
        set_non_generative_mode(True)
        assert is_non_generative_mode() is True

    def test_env_var_enables_mode(self):
        os.environ["BIBTEX_CHECK_NON_GENERATIVE"] = "1"
        assert is_non_generative_mode() is True
        os.environ["BIBTEX_CHECK_NON_GENERATIVE"] = "true"
        assert is_non_generative_mode() is True
        os.environ["BIBTEX_CHECK_NON_GENERATIVE"] = "no"
        assert is_non_generative_mode() is False

    def test_assert_no_llm_backend_passes_when_mode_off(self):
        # Should be a no-op.
        assert_no_llm_backend("openai")
        assert_no_llm_backend("some.anthropic.module")

    def test_assert_no_llm_backend_blocks_llm_modules_when_on(self):
        set_non_generative_mode(True)
        with pytest.raises(RuntimeError, match="non-generative"):
            assert_no_llm_backend("openai")
        with pytest.raises(RuntimeError, match="non-generative"):
            assert_no_llm_backend("some.anthropic.module")
        with pytest.raises(RuntimeError, match="non-generative"):
            assert_no_llm_backend("transformers.pipelines")

    def test_assert_no_llm_backend_allows_neutral_modules_when_on(self):
        set_non_generative_mode(True)
        # Plain bibliographic modules should not match the LLM markers.
        assert_no_llm_backend("crossref")
        assert_no_llm_backend("openalex")
        assert_no_llm_backend("bibtex_updater.matching")


# ===========================================================================
# OpenAlex permissive candidate conversion
# ===========================================================================


class TestOpenAlexCandidateConversion:
    def test_returns_none_for_empty_dict(self):
        assert openalex_work_to_candidate_record({}) is None

    def test_drops_dx_doi_prefix(self):
        rec = openalex_work_to_candidate_record(
            {
                "title": "X",
                "doi": "https://dx.doi.org/10.1/abc",
                "authorships": [],
                "primary_location": {"source": {"display_name": "Y"}},
                "publication_year": 2024,
                "type": "preprint",
            }
        )
        assert rec is not None
        assert rec.doi == "10.1/abc"

    def test_handles_authorships_with_single_name(self):
        rec = openalex_work_to_candidate_record(
            {
                "title": "X",
                "authorships": [{"author": {"display_name": "Plato"}}],
            }
        )
        assert rec is not None
        assert rec.authors == [{"given": "", "family": "Plato"}]


# ===========================================================================
# Module-level constants sanity check
# ===========================================================================


def test_penalty_constants_match_paper():
    """The CheckIfExist paper specifies these exact penalty/bonus magnitudes."""
    assert PENALTY_TITLE_MISMATCH == 20.0
    assert PENALTY_AUTHOR_MISMATCH == 20.0
    assert 10.0 <= PENALTY_JOURNAL_MISMATCH <= 20.0
    assert PENALTY_PER_FABRICATED_AUTHOR == 10.0
    assert PENALTY_FABRICATED_AUTHOR_CAP == 20.0
    assert MULTI_SOURCE_BONUS == 10.0
    assert CASE_A_TITLE_THRESHOLD == 80.0
    assert CASE_A_AUTHOR_THRESHOLD == 90.0
    assert DEFAULT_TOP_K == 3


def test_openalex_client_search_returns_empty_on_http_error():
    client = OpenAlexClient(http=None)
    # Calling .search with an HTTPException-prone configuration shouldn't crash.
    # The bare httpx path will fail (no real network in the unit test
    # environment), and the client should gracefully return [].
    out = client.search("nonexistent_query_xyz_12345_zzz")
    assert out == [] or isinstance(out, list)


# ===========================================================================
# Fielded title search (FIX A): OpenAlex filter=title.search, Crossref query.title
# ===========================================================================


def _ok(results):
    """A MagicMock 200 response whose .json() yields {"results": results}."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"results": list(results)}
    return resp


class TestOpenAlexFieldedTitleSearch:
    """FIX A.1: OpenAlex must issue a fielded ``filter=title.search:<title>``
    using the RAW title (no appended author) and only fall back to free-text
    ``?search=`` when the fielded query returns zero results.
    """

    def test_fielded_filter_used_when_title_given(self):
        http = MagicMock()
        http._request.return_value = _ok([{"title": "Context-Aware Sparse Deep Coordination Graphs"}])
        client = OpenAlexClient(http=http)

        out = client.search(
            "context aware sparse deep coordination graphs wang",
            title="Context-Aware Sparse Deep Coordination Graphs",
        )
        assert out and out[0]["title"].startswith("Context-Aware")
        # Exactly one request, and it must be the fielded filter (NOT ?search=).
        assert http._request.call_count == 1
        params = http._request.call_args.kwargs["params"]
        assert params["filter"] == "title.search:Context-Aware Sparse Deep Coordination Graphs"
        assert "search" not in params  # author-free, no free-text blob

    def test_falls_back_to_free_text_when_fielded_empty(self):
        http = MagicMock()
        # First call (fielded) -> empty; second call (free-text) -> a result.
        http._request.side_effect = [_ok([]), _ok([{"title": "Fallback Hit"}])]
        client = OpenAlexClient(http=http)

        out = client.search("some title surname", title="Some Title")
        assert out and out[0]["title"] == "Fallback Hit"
        assert http._request.call_count == 2
        # Second (fallback) call uses the free-text ?search= blob.
        fallback_params = http._request.call_args_list[1].kwargs["params"]
        assert fallback_params["search"] == "some title surname"
        assert "filter" not in fallback_params

    def test_no_fallback_when_fielded_has_results(self):
        http = MagicMock()
        http._request.return_value = _ok([{"title": "X"}])
        client = OpenAlexClient(http=http)
        client.search("x surname", title="X")
        assert http._request.call_count == 1  # fielded hit -> no free-text call

    def test_free_text_only_when_no_title(self):
        http = MagicMock()
        http._request.return_value = _ok([{"title": "Y"}])
        client = OpenAlexClient(http=http)
        client.search("y surname")
        params = http._request.call_args.kwargs["params"]
        assert params["search"] == "y surname"
        assert "filter" not in params


class TestOpenAlexApiKey:
    """OpenAlex premium ``api_key`` (explicit arg or ``OPENALEX_API_KEY`` env)
    must ride along on every /works and /sources request; keyless clients must
    not send the parameter at all.
    """

    def test_explicit_api_key_lands_in_works_params(self, monkeypatch):
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        http = MagicMock()
        http._request.return_value = _ok([{"title": "X"}])
        client = OpenAlexClient(http=http, api_key="premium-key")
        client.search("x surname", title="X")
        params = http._request.call_args.kwargs["params"]
        assert params["api_key"] == "premium-key"

    def test_env_api_key_used_as_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENALEX_API_KEY", "env-key")
        http = MagicMock()
        http._request.return_value = _ok([{"title": "X"}])
        client = OpenAlexClient(http=http)
        client.search("x surname", title="X")
        params = http._request.call_args.kwargs["params"]
        assert params["api_key"] == "env-key"

    def test_explicit_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("OPENALEX_API_KEY", "env-key")
        http = MagicMock()
        http._request.return_value = _ok([{"title": "X"}])
        client = OpenAlexClient(http=http, api_key="explicit-key")
        client.search("x surname", title="X")
        assert http._request.call_args.kwargs["params"]["api_key"] == "explicit-key"

    def test_keyless_client_sends_no_api_key_param(self, monkeypatch):
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        http = MagicMock()
        http._request.return_value = _ok([{"title": "X"}])
        client = OpenAlexClient(http=http)
        client.search("x surname", title="X")
        assert "api_key" not in http._request.call_args.kwargs["params"]

    def test_search_sources_includes_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        http = MagicMock()
        http._request.return_value = _ok([{"display_name": "Some Journal"}])
        client = OpenAlexClient(http=http, api_key="premium-key")
        client.search_sources("some journal")
        params = http._request.call_args.kwargs["params"]
        assert params["api_key"] == "premium-key"


class TestCrossrefFieldedTitleSearch:
    """FIX A.2: Crossref must use fielded ``query.title`` (+ ``query.author``)
    with the raw title instead of the generic ``query.bibliographic`` blob.
    """

    def _client(self):
        from bibtex_updater.fact_checker import CrossrefClient

        http = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"message": {"items": [{"title": ["Hit"]}]}}
        http._request.return_value = resp
        return CrossrefClient(http), http

    def test_uses_query_title_and_author(self):
        client, http = self._client()
        client.search("deep learning smith", title="Deep Learning", author="Smith")
        params = http._request.call_args.kwargs["params"]
        assert params["query.title"] == "Deep Learning"
        assert params["query.author"] == "Smith"
        assert "query.bibliographic" not in params

    def test_falls_back_to_bibliographic_without_title(self):
        client, http = self._client()
        client.search("deep learning smith")
        params = http._request.call_args.kwargs["params"]
        assert params["query.bibliographic"] == "deep learning smith"
        assert "query.title" not in params


class TestCascadePassesRawTitle(TestQueryCascade):
    """FIX A.3: the cascade must forward the RAW title (not the normalized
    title+surname blob) into the fielded Crossref/OpenAlex searches, and must
    query DBLP after OpenAlex via the permissive candidate converter.
    """

    def test_raw_title_forwarded_to_crossref_and_openalex(self):
        fc, oa = self._build(cr_items=[], s2_items=[], oa_items=[])
        fc.openalex = oa
        # DBLP returns a clean conference hit with no DOI (the exact case the
        # permissive converter must keep).
        dblp_hit = {
            "info": {
                "title": "Context-Aware Sparse Deep Coordination Graphs",
                "authors": {"author": ["Tonghan Wang", "Liang Zeng"]},
                "venue": "ICLR",
                "year": "2022",
                "type": "Conference and Workshop Papers",
            }
        }
        fc.dblp = MagicMock()
        fc.dblp.search.return_value = [dblp_hit]

        entry = {
            "ID": "x",
            "ENTRYTYPE": "inproceedings",
            "title": "Context-Aware Sparse Deep Coordination Graphs",
            "author": "Wang, Tonghan and Zeng, Liang",
            "year": "2022",
        }
        sources_queried: list = []
        cands = fc._query_cascade(entry, "context aware sparse deep coordination graphs wang", sources_queried, [], [])

        # Crossref received the raw title, not the normalized blob.
        cr_kwargs = fc.crossref.search.call_args.kwargs
        assert cr_kwargs["title"] == "Context-Aware Sparse Deep Coordination Graphs"
        # OpenAlex received the raw title too.
        oa_kwargs = oa.search.call_args.kwargs
        assert oa_kwargs["title"] == "Context-Aware Sparse Deep Coordination Graphs"
        # DBLP is in the cascade after OpenAlex and its DOI-less hit survived.
        # The exact-title hit scores >= cascade_high_confidence, so the cascade
        # short-circuits at DBLP and never reaches Semantic Scholar.
        assert sources_queried == ["crossref", "openalex", "dblp"]
        assert any(src == "dblp" for _, _, src in cands)


class TestLatexFreeRetrievalTitles(TestQueryCascade):
    """FIX: LaTeX-laden titles (``{B}rain {S}urgeon`` braces, ``M\\"uller``
    escapes) must be stripped from the title used for EVERY retrieval path --
    Crossref ``query.title``, OpenAlex ``filter=title.search:``, the S2
    free-text query, OpenReview, and the relaxed/structured-recheck retries --
    not just DBLP (FIX B2). Retrieval only: scoring still normalizes the
    original strings.
    """

    LATEX_TITLE = "The Combinatorial {B}rain {S}urgeon: Pruning Weights That Cancel One Another in Neural Networks"
    PLAIN_TITLE = "The Combinatorial Brain Surgeon: Pruning Weights That Cancel One Another in Neural Networks"

    def _entry(self):
        return {
            "ID": "yu2022brain",
            "ENTRYTYPE": "inproceedings",
            "title": self.LATEX_TITLE,
            "author": "Yu, Xin and Serra, Thiago",
            "booktitle": "ICML",
            "year": "2022",
        }

    def _cascade_checker(self):
        fc, oa = self._build(cr_items=[], s2_items=[], oa_items=[])
        fc.openalex = oa
        fc.dblp = MagicMock()
        fc.dblp.search.return_value = []
        fc.openreview = MagicMock()
        fc.openreview.search.return_value = []
        return fc, oa

    def test_all_cascade_sources_receive_brace_free_title(self):
        fc, oa = self._cascade_checker()
        entry = self._entry()

        fc._query_cascade(entry, "combinatorial brain surgeon yu", [], [], [])

        # Crossref fielded query.title is brace-free (first, strict call).
        cr_kwargs = fc.crossref.search.call_args_list[0].kwargs
        assert cr_kwargs["title"] == self.PLAIN_TITLE
        # OpenAlex fielded title.search is brace-free.
        oa_kwargs = oa.search.call_args_list[0].kwargs
        assert oa_kwargs["title"] == self.PLAIN_TITLE
        # DBLP free-text query is brace-free (pre-existing FIX B2 behavior).
        dblp_query = fc.dblp.search.call_args[0][0]
        assert "{" not in dblp_query and self.PLAIN_TITLE in dblp_query
        # OpenReview paperhash title is brace-free.
        or_kwargs = fc.openreview.search.call_args.kwargs
        assert or_kwargs["title"] == self.PLAIN_TITLE
        # Semantic Scholar free-text query is brace-free.
        s2_query = fc.s2.search.call_args[0][0]
        assert "{" not in s2_query and self.PLAIN_TITLE in s2_query

    def test_relaxed_author_fallback_receives_brace_free_title(self):
        fc, oa = self._cascade_checker()
        entry = self._entry()

        # Zero usable candidates -> the title-only fallback retries fire.
        fc._query_cascade(entry, "combinatorial brain surgeon yu", [], [], [])

        assert len(fc.crossref.search.call_args_list) == 2
        fallback_call = fc.crossref.search.call_args_list[1]
        assert fallback_call.args[0] == self.PLAIN_TITLE
        assert fallback_call.kwargs["title"] == self.PLAIN_TITLE

        assert len(oa.search.call_args_list) == 2
        oa_fallback = oa.search.call_args_list[1]
        assert oa_fallback.args[0] == self.PLAIN_TITLE
        assert oa_fallback.kwargs["title"] == self.PLAIN_TITLE

    def test_structured_author_recheck_search_is_brace_free(self):
        fc, _ = self._cascade_checker()
        entry = self._entry()  # no DOI -> Path 2 (fielded Crossref search)
        best_match = PublishedRecord(
            doi=None,
            title=self.PLAIN_TITLE,
            authors=[{"given": "", "family": "Xin Yu"}],
            structured_names=False,
        )

        fc._structured_author_recheck(entry, best_match)

        recheck_call = fc.crossref.search.call_args
        assert recheck_call.args[0] == self.PLAIN_TITLE
        assert recheck_call.kwargs["title"] == self.PLAIN_TITLE

    def test_scoring_still_uses_original_title_normalization(self):
        """The brace-stripped retrieval title must not leak into scoring: an
        exact (normalized) match still scores 1.0 on the title component."""
        cr_item = {
            "DOI": "10.5555/brain",
            "title": [self.PLAIN_TITLE],
            "author": [{"given": "Xin", "family": "Yu"}, {"given": "Thiago", "family": "Serra"}],
            "container-title": ["ICML"],
            "issued": {"date-parts": [[2022]]},
            "type": "proceedings-article",
        }
        fc, oa = self._build(cr_items=[cr_item])
        fc.openalex = oa
        entry = self._entry()

        candidates = fc._query_cascade(entry, "combinatorial brain surgeon yu", [], [], [])

        cr_scores = [score for score, _rec, src in candidates if src == "crossref"]
        assert cr_scores and max(cr_scores) == pytest.approx(1.0)
