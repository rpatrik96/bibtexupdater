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
    build_openreview_paperhashes,
    openreview_note_to_candidate_record,
)
from bibtex_updater.utils import (
    OPENREVIEW_API,
    OPENREVIEW_API_V2,
    RateLimiterRegistry,
    SourceUnavailableError,
)

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


class TestOpenReviewPaperhashMatchesTheLiveIndex:
    """Every expectation here was read off the live index under an
    authenticated session, by issuing the hash and counting the notes returned.
    The pre-fix normalizer produced the "misses" column and got 0 notes.
    """

    @pytest.mark.parametrize(
        "title, author, expected",
        [
            # Latin-1 is PRESERVED. "akyürek|..." returns 2 notes on v1;
            # "akyurek|..." returns 0.
            (
                "What Learning Algorithm Is In-Context Learning? Investigations with Linear Models",
                'Aky{\\"u}rek, Ekin',
                "akyürek|what_learning_algorithm_is_incontext_learning_investigations_with_linear_models",
            ),
            (
                "Nonparametric Identifiability of Causal Representations from Unknown Interventions",
                'von K{\\"u}gelgen, Julius',
                "kügelgen|nonparametric_identifiability_of_causal_representations_from_unknown_interventions",
            ),
            # Latin Extended is DROPPED, not folded: "Karlaš" indexes as
            # "karla" (1 note on v2); the folded "karlas" is a DIFFERENT note
            # (the DBLP mirror), which is why both forms are issued.
            (
                "Data Debugging with Shapley Importance over Machine Learning Pipelines",
                "Karla\u0161, Bojan",
                "karla|data_debugging_with_shapley_importance_over_machine_learning_pipelines",
            ),
            # Maths survives into the title key: the backslash, the underscore
            # and the caret are all part of it (2 notes on v2 for the first,
            # 0 for the punctuation-stripped form).
            (
                "Rethinking 3D Convolution in $\\ell_p$-norm Space",
                "Li Zhang",
                "zhang|rethinking_3d_convolution_in_\\ell_pnorm_space",
            ),
            (
                "Optimality of Matrix Mechanism on $\\ell_p^p$-metric",
                "Zongrui Zou",
                "zou|optimality_of_matrix_mechanism_on_\\ell_p^pmetric",
            ),
            # The surname is the LAST whitespace token; nobiliary particles go
            # with the rest of the name. "morvan|..." returns 2 notes on v1,
            # "le_morvan|..." returns 0.
            (
                "NeuMiss networks: differentiable programming for supervised learning with missing values",
                "Le Morvan, Marine",
                "morvan|neumiss_networks_differentiable_programming_for_supervised_learning_with_missing_values",
            ),
            (
                "Representation Learning with Contrastive Predictive Coding",
                "van den Oord, Aaron",
                "oord|representation_learning_with_contrastive_predictive_coding",
            ),
            # Punctuation inside a surname is deleted, not spaced, so O'Brien
            # stays one token.
            (
                "A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy",
                "O'Brien, Kyle",
                "obrien|a_few_bad_neurons_isolating_and_surgically_correcting_sycophancy",
            ),
        ],
    )
    def test_reproduces_a_live_hash(self, title, author, expected):
        assert expected in build_openreview_paperhashes(title, author)

    def test_issues_both_diacritic_forms(self):
        assert build_openreview_paperhashes("A Title", 'Aky{\\"u}rek, Ekin') == [
            "akyürek|a_title",
            "akyurek|a_title",
        ]

    def test_collapses_to_one_form_when_unaccented(self):
        assert build_openreview_paperhashes("A Title", "Kingma, Diederik P.") == ["kingma|a_title"]

    def test_accepts_an_already_reduced_surname(self):
        # The cascade used to hand the client a folded, reduced surname; that
        # input must still hash the same way.
        assert build_openreview_paperhashes("A Title", "kingma") == ["kingma|a_title"]

    def test_empty_when_unhashable(self):
        assert build_openreview_paperhashes("", "kingma") == []
        assert build_openreview_paperhashes("A Title", "") == []


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

    def test_raises_on_exception(self):
        http = MagicMock()
        http._request.side_effect = RuntimeError("network down")
        client = OpenReviewClient(http=http)
        # A quiet [] here made a dead network look like "OpenReview does not
        # host this paper", which is the claim not_found rests on.
        with pytest.raises(SourceUnavailableError):
            client.search("b", title="T", first_author="a")

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


class TestOpenReviewTwoHostPaperhash:
    """The two hosts are DISJOINT, so the paperhash lookup runs against both.

    Counted live under an authenticated session: ICLR 2024 has 0 notes on v1
    and 2,260 on v2, NeurIPS 2024 0 and 4,035, TMLR 0 and 4,639; ICLR 2021 has
    860 on v1 and 0 on v2. A v1-only paperhash lookup can never confirm a
    2023-or-later paper, which is the dominant shape in a modern ML
    bibliography.
    """

    def test_v2_is_queried_first(self):
        # Before the fix the single paperhash request went to v1.
        http = MagicMock()
        http._request.return_value = _ok([_v2_note("Adam", ["D Kingma"], ["~Diederik_Kingma1"])])
        client = OpenReviewClient(http=http)

        out = client.search("blob", title="Adam", first_author="kingma")

        assert out
        assert http._request.call_count == 1
        assert http._request.call_args.args[1] == f"{OPENREVIEW_API_V2}/notes"

    def test_v2_only_venue_found_where_v1_misses(self):
        # The regression this whole change exists for: a 2024 ICLR paper is
        # invisible on v1 and present on v2. v1 answers "no notes", v2 answers
        # with the paper.
        v2_note = _v2_note(
            "Sparse Attention Revisited",
            ["Grace Hopper", "Alan Turing"],
            ["~Grace_Hopper1", "~Alan_Turing1"],
            venue="ICLR 2024 poster",
            venueid="ICLR.cc/2024/Conference",
        )

        def by_host(method, url, **kwargs):
            return _ok([v2_note]) if url.startswith(OPENREVIEW_API_V2) else _ok([])

        http = MagicMock()
        http._request.side_effect = by_host
        client = OpenReviewClient(http=http)

        out = client.search("blob", limit=3, title="Sparse Attention Revisited", first_author="hopper")

        assert out == [v2_note]
        call = http._request.call_args_list[0]
        assert call.args[1] == f"{OPENREVIEW_API_V2}/notes"
        assert call.kwargs["service"] == "openreview"
        assert call.kwargs["params"] == {"paperhash": "hopper|sparse_attention_revisited", "limit": 3}

        rec = openreview_note_to_candidate_record(out[0])
        assert rec is not None
        assert rec.journal == "ICLR 2024 poster"
        assert rec.year == 2024  # recovered from the venue string
        assert rec.structured_names is True
        assert rec.surname_keys() == ["hopper", "turing"]

    def test_v1_queried_when_v2_misses(self):
        # A pre-2023 venue lives on v1 alone, so a v2 miss is not the end.
        v1_note = _v1_note("Adam", ["D Kingma"], ["~Diederik_Kingma1"], venue="ICLR 2015")
        http = MagicMock()
        http._request.side_effect = [_ok([]), _ok([v1_note])]
        client = OpenReviewClient(http=http)

        out = client.search("blob", title="Adam", first_author="kingma")

        assert out == [v1_note]
        urls = [c.args[1] for c in http._request.call_args_list]
        assert urls == [f"{OPENREVIEW_API_V2}/notes", f"{OPENREVIEW_API}/notes"]

    def test_both_hash_forms_are_issued(self):
        # OpenReview keeps Latin-1 and drops Latin Extended, and the DBLP mirror
        # of the same paper arrives already transliterated. A client cannot tell
        # which form a name lands in, so it asks for both.
        http = MagicMock()
        http._request.return_value = _ok([])
        client = OpenReviewClient(http=http)

        client.search("blob", title="A Title", first_author="Karla\u0161, Bojan")

        hashes = [
            c.kwargs["params"]["paperhash"] for c in http._request.call_args_list if "paperhash" in c.kwargs["params"]
        ]
        assert hashes[:2] == ["karla|a_title", "karlas|a_title"]
        # Both hosts get both forms before the miss is called exhaustive.
        assert hashes == ["karla|a_title", "karlas|a_title"] * 2

    def test_single_hash_when_name_is_unaccented(self):
        # The common case must not double the request budget.
        http = MagicMock()
        http._request.return_value = _ok([])
        client = OpenReviewClient(http=http)

        client.search("blob", title="A Title", first_author="Kingma, Diederik P.")

        hashes = [
            c.kwargs["params"]["paperhash"] for c in http._request.call_args_list if "paperhash" in c.kwargs["params"]
        ]
        assert hashes == ["kingma|a_title", "kingma|a_title"]

    def test_term_fallback_runs_on_both_hosts_after_a_double_miss(self):
        http = MagicMock()
        http._request.side_effect = [_ok([]), _ok([]), _ok([]), _ok([_v1_note("T U V", ["A B"], ["~A_B1"])])]
        client = OpenReviewClient(http=http)

        out = client.search("blob", limit=3, title="T U V", first_author="a")

        assert out
        urls = [c.args[1] for c in http._request.call_args_list]
        assert urls == [
            f"{OPENREVIEW_API_V2}/notes",
            f"{OPENREVIEW_API}/notes",
            f"{OPENREVIEW_API_V2}/notes/search",
            f"{OPENREVIEW_API}/notes/search",
        ]
        assert http._request.call_args_list[2].kwargs["params"] == {"term": "T U V", "limit": 3}

    def test_a_host_that_never_answered_raises(self):
        # One host answered "nothing"; the other never answered at all, so the
        # miss is not exhaustive and cannot support ``not_found``.
        def by_host(method, url, **kwargs):
            if url.startswith(OPENREVIEW_API_V2):
                raise RuntimeError("v2 down")
            return _ok([])

        http = MagicMock()
        http._request.side_effect = by_host
        client = OpenReviewClient(http=http)
        with pytest.raises(SourceUnavailableError):
            client.search("b", title="T U V", first_author="a")

    def test_a_hit_on_the_healthy_host_survives_the_other_failing(self):
        v2_note = _v2_note("T U V", ["A B"], ["~A_B1"], venue="ICLR 2024")

        def by_host(method, url, **kwargs):
            if url.startswith(OPENREVIEW_API_V2):
                return _ok([v2_note])
            raise RuntimeError("v1 down")

        http = MagicMock()
        http._request.side_effect = by_host
        client = OpenReviewClient(http=http)
        assert client.search("b", title="T U V", first_author="a") == [v2_note]

    def test_v2_non_200_raises(self):
        resp_500 = MagicMock()
        resp_500.status_code = 500
        http = MagicMock()
        http._request.side_effect = [resp_500, _ok([])]
        client = OpenReviewClient(http=http)
        with pytest.raises(SourceUnavailableError):
            client.search("b", title="T U V", first_author="a")

    def test_v2_malformed_notes_returns_empty(self):
        bad = MagicMock()
        bad.status_code = 200
        bad.json.return_value = {"notes": "not-a-list"}
        http = MagicMock()
        http._request.side_effect = [bad, _ok([]), _ok([]), _ok([])]
        client = OpenReviewClient(http=http)
        assert client.search("b", title="T U V", first_author="a") == []

    def test_no_request_without_first_author(self):
        # Same gating as before: author-less searches never reach any path.
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
        # OpenReview gets the RAW title and the RAW first-author name: the
        # shared retrieval forms delete maths and ASCII-fold the surname, and
        # OpenReview's paperhash index keeps both.
        kwargs = openreview.search.call_args.kwargs
        assert kwargs["title"] == "Some ICLR Paper"
        assert kwargs["first_author"] == "Doe, Jane"

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
