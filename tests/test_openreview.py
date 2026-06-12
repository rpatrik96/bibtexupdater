"""Hermetic tests for the OpenReview cascade source.

OpenReview (legacy ``api.openreview.net/notes?paperhash=...``) is the
authoritative submission registry for ICLR/NeurIPS/TMLR and many other ML
venues that the rest of the cascade frequently fails to *positively* confirm.
These tests mock the shared HTTP client -- no live network calls.

Covered:
- ``build_openreview_paperhash`` normalization (verified live against the
  Kingma/Vaswani/Devlin/Brown papers).
- ``OpenReviewClient.search`` request shape + error handling.
- ``openreview_note_to_candidate_record`` (v1/v2 content shapes, authoritative
  family extraction from ``~Given_Family<N>`` handles, ``structured_names``).
- Cascade wiring: OpenReview sits after DBLP, before Semantic Scholar, and
  short-circuits at high confidence.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from bibtex_updater.fact_checker import (
    CASCADE_HIGH_CONFIDENCE,
    FactChecker,
    FactCheckerConfig,
)
from bibtex_updater.sources import (
    OpenReviewClient,
    _content_value,
    build_openreview_paperhash,
    openreview_note_to_candidate_record,
)
from bibtex_updater.utils import OPENREVIEW_API, OPENREVIEW_API_V2, RateLimiterRegistry

# ------------- Helpers -------------


def _ok(notes):
    """A MagicMock 200 response whose ``.json()`` yields ``{"notes": notes}``."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"notes": list(notes)}
    return resp


def _v2_note(title, authors, authorids, venue=None, venueid=None, year=None):
    """Build an OpenReview API-v2 note (every content field wrapped in value)."""
    content = {
        "title": {"value": title},
        "authors": {"value": list(authors)},
        "authorids": {"value": list(authorids)},
    }
    if venue is not None:
        content["venue"] = {"value": venue}
    if venueid is not None:
        content["venueid"] = {"value": venueid}
    if year is not None:
        content["year"] = {"value": year}
    return {"id": "note1", "content": content}


def _v1_note(title, authors, authorids, venue=None):
    """Build a legacy API-v1 note (bare content values)."""
    content = {"title": title, "authors": list(authors), "authorids": list(authorids)}
    if venue is not None:
        content["venue"] = venue
    return {"id": "note1", "content": content}


# ------------- paperhash construction -------------


class TestBuildOpenReviewPaperhash:
    @pytest.mark.parametrize(
        "title, last, expected",
        [
            # Colon dropped, spaces -> underscores (verified live).
            (
                "Adam: A Method for Stochastic Optimization",
                "kingma",
                "kingma|adam_a_method_for_stochastic_optimization",
            ),
            # Hyphen removed (NOT spaced): "Few-Shot" -> "fewshot".
            (
                "Language Models are Few-Shot Learners",
                "Brown",
                "brown|language_models_are_fewshot_learners",
            ),
            # Colon + hyphen both dropped.
            (
                "BERT: Pre-training of Deep Bidirectional Transformers",
                "Devlin",
                "devlin|bert_pretraining_of_deep_bidirectional_transformers",
            ),
        ],
    )
    def test_normalization(self, title, last, expected):
        assert build_openreview_paperhash(title, last) == expected

    def test_diacritics_preserved_in_author(self):
        # FIX B1: OpenReview's paperhash index keys on the Unicode-preserved
        # surname ("müller" vs "muller"); stripping diacritics here returned 0
        # notes for accented author names. Preserve diacritics so the paperhash
        # matches OpenReview's index.
        assert build_openreview_paperhash("A Title", "Müller") == "müller|a_title"

    def test_collapses_whitespace(self):
        assert build_openreview_paperhash("A   Spaced    Title", "x") == "x|a_spaced_title"

    def test_none_on_empty_title(self):
        assert build_openreview_paperhash("", "kingma") is None
        assert build_openreview_paperhash("   :::   ", "kingma") is None

    def test_none_on_empty_author(self):
        assert build_openreview_paperhash("Some Title", "") is None
        assert build_openreview_paperhash("Some Title", "  ") is None


# ------------- OpenReviewClient.search -------------


class TestOpenReviewClientSearch:
    def test_builds_paperhash_param(self):
        http = MagicMock()
        http._request.return_value = _ok([_v2_note("Adam: A Method", ["D Kingma"], ["~Diederik_Kingma1"])])
        client = OpenReviewClient(http=http)

        out = client.search("blob", limit=3, title="Adam: A Method", first_author="kingma")

        assert out and out[0]["id"] == "note1"
        assert http._request.call_count == 1
        call = http._request.call_args
        # Routed through the shared client with the openreview service tag.
        assert call.kwargs["service"] == "openreview"
        assert call.kwargs["params"]["paperhash"] == "kingma|adam_a_method"
        assert call.kwargs["params"]["limit"] == 3

    def test_no_request_without_title(self):
        http = MagicMock()
        client = OpenReviewClient(http=http)
        assert client.search("blob", title=None, first_author="kingma") == []
        http._request.assert_not_called()

    def test_no_request_without_author(self):
        http = MagicMock()
        client = OpenReviewClient(http=http)
        assert client.search("blob", title="Adam", first_author=None) == []
        http._request.assert_not_called()

    def test_empty_on_non_200(self):
        http = MagicMock()
        resp = MagicMock()
        resp.status_code = 404
        http._request.return_value = resp
        client = OpenReviewClient(http=http)
        assert client.search("b", title="T", first_author="a") == []

    def test_empty_on_exception(self):
        http = MagicMock()
        http._request.side_effect = RuntimeError("network down")
        client = OpenReviewClient(http=http)
        # Must never raise -- the cascade depends on a quiet [] on failure.
        assert client.search("b", title="T", first_author="a") == []

    def test_empty_on_malformed_json(self):
        http = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"notes": "not-a-list"}
        http._request.return_value = resp
        client = OpenReviewClient(http=http)
        assert client.search("b", title="T", first_author="a") == []

    def test_limit_capped_at_max(self):
        http = MagicMock()
        http._request.return_value = _ok([])
        client = OpenReviewClient(http=http)
        client.search("b", limit=999, title="T", first_author="a")
        assert http._request.call_args.kwargs["params"]["limit"] <= 10


# ------------- API v2 fallback -------------


class TestOpenReviewV2Fallback:
    """Venues that migrated to api2.openreview.net (ICLR 2024+, NeurIPS 2023+)
    are invisible to the legacy v1 host; when v1 paperhash AND v1 term both
    miss, the same term search runs against the v2 ``/notes/search`` endpoint.
    """

    def test_v1_paperhash_hit_skips_v2(self):
        # (1) A v1 paperhash hit must short-circuit: exactly one request, to v1.
        http = MagicMock()
        http._request.return_value = _ok([_v1_note("Adam", ["D Kingma"], ["~Diederik_Kingma1"])])
        client = OpenReviewClient(http=http)

        out = client.search("blob", title="Adam", first_author="kingma")

        assert out
        assert http._request.call_count == 1
        assert http._request.call_args.args[1] == f"{OPENREVIEW_API}/notes"

    def test_v1_term_hit_skips_v2(self):
        # v1 term fallback hit -> two requests, both v1, no v2 call.
        http = MagicMock()
        http._request.side_effect = [
            _ok([]),
            _ok([_v1_note("Adam", ["D Kingma"], ["~Diederik_Kingma1"])]),
        ]
        client = OpenReviewClient(http=http)

        out = client.search("blob", title="Adam", first_author="kingma")

        assert out
        assert http._request.call_count == 2
        for call in http._request.call_args_list:
            assert call.args[1] == f"{OPENREVIEW_API}/notes"

    def test_v1_double_miss_falls_back_to_v2(self):
        # (2) v1 paperhash miss + v1 term miss -> v2 /notes/search is queried
        # with the same LaTeX-stripped term, through the shared client with
        # the openreview service tag; the v2-shaped note parses correctly.
        v2_note = _v2_note(
            "Sparse {A}ttention Revisited",
            ["Grace Hopper", "Alan Turing"],
            ["~Grace_Hopper1", "~Alan_Turing1"],
            venue="ICLR 2024 poster",
            venueid="ICLR.cc/2024/Conference",
        )
        http = MagicMock()
        http._request.side_effect = [_ok([]), _ok([]), _ok([v2_note])]
        client = OpenReviewClient(http=http)

        out = client.search("blob", limit=3, title="Sparse {A}ttention Revisited", first_author="hopper")

        assert out == [v2_note]
        assert http._request.call_count == 3
        v2_call = http._request.call_args_list[2]
        assert v2_call.args[1] == f"{OPENREVIEW_API_V2}/notes/search"
        assert v2_call.kwargs["service"] == "openreview"
        # LaTeX-stripped term, same params shape as the v1 term fallback.
        assert v2_call.kwargs["params"] == {"term": "Sparse Attention Revisited", "limit": 3}

        rec = openreview_note_to_candidate_record(out[0])
        assert rec is not None
        assert rec.title == "Sparse {A}ttention Revisited"
        assert rec.journal == "ICLR 2024 poster"
        assert rec.year == 2024  # recovered from the venue string
        assert rec.structured_names is True  # families from the tilde handles
        assert rec.surname_keys() == ["hopper", "turing"]
        assert rec.order_reliable is True  # same as the v1 converter

    def test_v2_error_returns_empty(self):
        # (3) Any v2 failure -> [] (never raises into the cascade).
        http = MagicMock()
        http._request.side_effect = [_ok([]), _ok([]), RuntimeError("v2 down")]
        client = OpenReviewClient(http=http)
        assert client.search("b", title="T U V", first_author="a") == []

    def test_v2_non_200_returns_empty(self):
        http = MagicMock()
        resp_500 = MagicMock()
        resp_500.status_code = 500
        http._request.side_effect = [_ok([]), _ok([]), resp_500]
        client = OpenReviewClient(http=http)
        assert client.search("b", title="T U V", first_author="a") == []

    def test_v2_malformed_notes_returns_empty(self):
        bad = MagicMock()
        bad.status_code = 200
        bad.json.return_value = {"notes": "not-a-list"}
        http = MagicMock()
        http._request.side_effect = [_ok([]), _ok([]), bad]
        client = OpenReviewClient(http=http)
        assert client.search("b", title="T U V", first_author="a") == []

    def test_no_v2_without_first_author(self):
        # Same gating as the v1 term fallback: author-less searches never
        # reach the term/v2 paths at all.
        http = MagicMock()
        http._request.return_value = _ok([])
        client = OpenReviewClient(http=http)
        assert client.search("blob", title="Adam", first_author=None) == []
        http._request.assert_not_called()


# ------------- _content_value (v1/v2 shapes) -------------


class TestContentValue:
    def test_v1_bare_value(self):
        assert _content_value({"title": "Adam"}, "title") == "Adam"

    def test_v2_wrapped_value(self):
        assert _content_value({"title": {"value": "Adam"}}, "title") == "Adam"

    def test_v2_wrapped_list(self):
        assert _content_value({"authors": {"value": ["A", "B"]}}, "authors") == ["A", "B"]

    def test_missing_key_is_none(self):
        assert _content_value({}, "title") is None

    def test_v2_wrapped_none(self):
        assert _content_value({"title": {"value": None}}, "title") is None

    def test_dict_without_value_key_passes_through(self):
        raw = {"something": "else"}
        assert _content_value({"title": raw}, "title") is raw


# ------------- note -> PublishedRecord conversion -------------


class TestOpenReviewNoteToCandidateRecord:
    def test_v2_structured_authors_from_tilde_ids(self):
        note = _v2_note(
            "Adam: A Method for Stochastic Optimization",
            ["Diederik P. Kingma", "Jimmy Ba"],
            ["~Diederik_P_Kingma1", "~Jimmy_Ba1"],
            venue="ICLR (Poster) 2015",
            venueid="dblp.org/journals/CORR/2015",
        )
        rec = openreview_note_to_candidate_record(note)
        assert rec is not None
        assert rec.title == "Adam: A Method for Stochastic Optimization"
        assert rec.journal == "ICLR (Poster) 2015"
        # All authorids are tilde handles -> authoritative family names.
        assert rec.structured_names is True
        assert rec.surname_keys() == ["kingma", "ba"]
        assert rec.authors[0] == {"given": "Diederik P.", "family": "Kingma"}

    def test_v1_bare_content(self):
        note = _v1_note(
            "Auto-Encoding Variational Bayes",
            ["Diederik P. Kingma", "Max Welling"],
            ["~Diederik_P_Kingma1", "~Max_Welling1"],
            venue="ICLR 2014",
        )
        rec = openreview_note_to_candidate_record(note)
        assert rec is not None
        assert rec.title == "Auto-Encoding Variational Bayes"
        assert rec.journal == "ICLR 2014"
        assert rec.structured_names is True
        assert rec.surname_keys() == ["kingma", "welling"]

    def test_unstructured_when_authorid_not_tilde(self):
        # DBLP-search-URL authorids carry no family handle -> synthesize +
        # mark unstructured so the verdict logic stays conservative.
        note = _v2_note(
            "Attention Is All You Need",
            ["Ashish Vaswani", "Niki Parmar"],
            ["~Ashish_Vaswani1", "https://dblp.org/search/pid/api?q=author:Niki_Parmar:"],
            venue="CoRR 2017",
        )
        rec = openreview_note_to_candidate_record(note)
        assert rec is not None
        assert rec.structured_names is False
        # Synthesized family via last-token heuristic still yields right keys.
        assert rec.surname_keys() == ["vaswani", "parmar"]

    def test_family_from_tilde_with_middle_initial_dot(self):
        note = _v2_note("X", ["Aidan N. Gomez"], ["~Aidan_N._Gomez1"])
        rec = openreview_note_to_candidate_record(note)
        assert rec.surname_keys() == ["gomez"]
        assert rec.structured_names is True

    def test_venueid_fallback_when_no_venue(self):
        note = _v2_note("X", ["A B"], ["~A_B1"], venueid="ICLR.cc/2024/Conference")
        rec = openreview_note_to_candidate_record(note)
        assert rec.journal == "ICLR.cc/2024/Conference"

    def test_year_parsed_from_content(self):
        note = _v2_note("X", ["A B"], ["~A_B1"], year="2020")
        rec = openreview_note_to_candidate_record(note)
        assert rec.year == 2020

    def test_none_on_no_title(self):
        note = _v2_note("", ["A B"], ["~A_B1"])
        assert openreview_note_to_candidate_record(note) is None

    def test_none_on_empty_note(self):
        assert openreview_note_to_candidate_record({}) is None
        assert openreview_note_to_candidate_record(None) is None

    def test_strips_html_in_title(self):
        note = _v2_note("Deep <i>Learning</i>", ["A B"], ["~A_B1"])
        rec = openreview_note_to_candidate_record(note)
        assert rec.title == "Deep Learning"

    def test_no_doi(self):
        rec = openreview_note_to_candidate_record(_v2_note("X", ["A B"], ["~A_B1"]))
        assert rec.doi is None


# ------------- cascade wiring -------------


class TestOpenReviewCascadeWiring:
    """OpenReview sits AFTER DBLP and BEFORE Semantic Scholar."""

    def _build(self, openreview):
        crossref = MagicMock()
        crossref.search.return_value = []
        crossref.http = MagicMock()
        dblp = MagicMock()
        dblp.search.return_value = []
        s2 = MagicMock()
        s2.search.return_value = []
        openalex = MagicMock()
        openalex.search.return_value = []
        fc = FactChecker(
            crossref,
            dblp,
            s2,
            FactCheckerConfig(top_k=3),
            logging.getLogger("openreview-test"),
            openalex=openalex,
            openreview=openreview,
        )
        return fc, s2

    def test_openreview_queried_after_dblp_before_s2(self):
        openreview = MagicMock()
        openreview.search.return_value = []
        fc, _ = self._build(openreview)
        entry = {
            "ID": "x",
            "ENTRYTYPE": "inproceedings",
            "title": "Some ICLR Paper",
            "author": "Doe, Jane",
            "year": "2024",
        }
        sources_queried: list = []
        fc._query_cascade(entry, "Some ICLR Paper Doe", sources_queried, [], [])
        # FIX X4: when every primary source returns nothing usable, the
        # relaxed-author retrieval fallback runs (title-only retry on
        # Crossref + OpenAlex) and appends two fallback source names.
        assert sources_queried == [
            "crossref",
            "openalex",
            "dblp",
            "openreview",
            "semanticscholar",
            "crossref-fallback",
            "openalex-fallback",
        ]
        # OpenReview got the raw title + reduced first-author surname.
        kwargs = openreview.search.call_args.kwargs
        assert kwargs["title"] == "Some ICLR Paper"
        assert kwargs["first_author"] == "doe"

    def test_high_confidence_openreview_short_circuits_before_s2(self):
        title = "A Highly Specific ICLR Submission Title"
        openreview = MagicMock()
        openreview.search.return_value = [
            _v2_note(
                title,
                ["Jane Doe", "John Roe"],
                ["~Jane_Doe1", "~John_Roe1"],
                venue="ICLR 2024",
            )
        ]
        fc, s2 = self._build(openreview)
        entry = {
            "ID": "x",
            "ENTRYTYPE": "inproceedings",
            "title": title,
            "author": "Doe, Jane and Roe, John",
            "year": "2024",
        }
        sources_queried: list = []
        cands = fc._query_cascade(entry, f"{title} Doe", sources_queried, [], [])
        # Exact title + matching authors -> >= high-confidence at OpenReview.
        assert any(score >= CASCADE_HIGH_CONFIDENCE and src == "openreview" for score, _, src in cands)
        assert "openreview" in sources_queried
        # Short-circuited: Semantic Scholar never reached.
        assert "semanticscholar" not in sources_queried
        s2.search.assert_not_called()

    def test_lazily_built_from_shared_http_when_none(self):
        crossref = MagicMock()
        crossref.search.return_value = []
        crossref.http = MagicMock()
        crossref.http._request.return_value = _ok([])
        dblp = MagicMock()
        dblp.search.return_value = []
        s2 = MagicMock()
        s2.search.return_value = []
        openalex = MagicMock()
        openalex.search.return_value = []
        fc = FactChecker(
            crossref,
            dblp,
            s2,
            FactCheckerConfig(top_k=3),
            logging.getLogger("openreview-lazy"),
            openalex=openalex,
        )
        assert fc.openreview is None
        entry = {"ID": "x", "ENTRYTYPE": "article", "title": "T", "author": "Doe, Jane"}
        sources_queried: list = []
        fc._query_cascade(entry, "T Doe", sources_queried, [], [])
        # Lazily constructed and queried via the shared crossref.http.
        assert isinstance(fc.openreview, OpenReviewClient)
        assert "openreview" in sources_queried


# ------------- rate-limit registry -------------


def test_openreview_in_default_rate_limits():
    assert "openreview" in RateLimiterRegistry.DEFAULT_LIMITS
    assert RateLimiterRegistry.DEFAULT_LIMITS["openreview"] == 30
    # And the limiter is constructible for the service.
    assert RateLimiterRegistry().get("openreview") is not None
