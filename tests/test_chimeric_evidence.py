"""Auditability of the chimeric-title HALLUCINATED verdict (issue #70).

``_detect_chimeric_title`` used to return a bare bool, and ``check_entry``
turned it into HALLUCINATED at a hard-coded 0.95 with every evidence field
empty. The detection rule and its thresholds (>= 4 shared tokens per source,
>= 3 tokens unique to each) are unchanged; what these tests pin down is that
the verdict now carries the evidence behind it -- the two sources, their
titles, the token sets -- and that the confidence is a measurement of the
margin over the thresholds rather than a constant.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from bibtex_updater.fact_checker import (
    CHIMERIC_CONFIDENCE_CEILING,
    CHIMERIC_CONFIDENCE_FLOOR,
    CHIMERIC_MIN_SHARED_TOKENS,
    CHIMERIC_MIN_UNIQUE_TOKENS,
    ChimericEvidence,
    CrossrefClient,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    FactCheckProcessor,
    FactCheckResult,
    FactCheckStatus,
    FieldComparison,
    SemanticScholarClient,
    build_verification_result,
    chimeric_confidence,
)
from bibtex_updater.utils import PublishedRecord

# ------------- Fixtures -------------


@pytest.fixture
def logger():
    return logging.getLogger("test_chimeric_evidence")


@pytest.fixture
def fact_checker(logger):
    """FactChecker whose HTTP layer answers nothing; the cascade is stubbed per test."""
    http = MagicMock()
    http._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return FactChecker(CrossrefClient(http), DBLPClient(http), SemanticScholarClient(http), FactCheckerConfig(), logger)


@pytest.fixture
def processor(fact_checker, logger):
    return FactCheckProcessor(fact_checker, logger)


#: "Attention Is All You Need" spliced onto a second real-looking title. After
#: stopword removal ("is", "in") the first paper contributes
#: {attention, all, you, need} (4 shared / 4 unique) and the second
#: {sparse, autoencoder, feature, interpretability, language, systems}
#: (6 shared / 6 unique).
SPLICED_ENTRY = {
    "ID": "chimera2017",
    "ENTRYTYPE": "article",
    "title": "Attention Is All You Need Sparse Autoencoder Feature Interpretability in Language Systems",
    "author": "Vaswani, Ashish",
    "year": "2017",
}

REC_ATTENTION = PublishedRecord(
    doi="10.5555/attention",
    title="Attention Is All You Need",
    authors=[{"given": "Ashish", "family": "Vaswani"}],
    journal="NeurIPS",
    year=2017,
)

REC_SAE = PublishedRecord(
    doi="10.5555/sae",
    title="Sparse Autoencoder Feature Interpretability in Language Systems",
    authors=[{"given": "Jane", "family": "Doe"}],
    journal="ICLR",
    year=2024,
)


def _stub_cascade(fact_checker, monkeypatch, records: list[tuple[PublishedRecord, str]]):
    """Make the cascade return ``records`` as (score, record, source) candidates."""

    def fake_cascade(entry, query, sq, sh, errors, failed=None):
        for _rec, source in records:
            if source not in sq:
                sq.append(source)
                sh.append(source)
        authors_ref = [entry["author"].split(",")[0].lower()] if entry.get("author") else []
        return [(fact_checker._score_candidate(query, authors_ref, rec), rec, source) for rec, source in records]

    monkeypatch.setattr(fact_checker, "_query_cascade", fake_cascade)
    monkeypatch.setattr(fact_checker, "_query_arxiv_by_id", lambda *a, **k: [])


def _spliced_result(fact_checker, monkeypatch) -> FactCheckResult:
    _stub_cascade(fact_checker, monkeypatch, [(REC_ATTENTION, "crossref"), (REC_SAE, "dblp")])
    return fact_checker.check_entry(dict(SPLICED_ENTRY))


def _candidates(*specs: tuple[str, str, float]) -> list[tuple[float, PublishedRecord, str]]:
    """Build detector candidates from (source, title, score) specs."""
    return [(score, PublishedRecord(doi=f"10.1/{source}", title=title), source) for source, title, score in specs]


# ------------- The verdict carries its evidence -------------


class TestSplicedTitleVerdict:
    def test_status_is_hallucinated_with_populated_record(self, fact_checker, monkeypatch):
        result = _spliced_result(fact_checker, monkeypatch)

        assert result.status is FactCheckStatus.HALLUCINATED
        assert result.chimeric_evidence is not None
        assert result.best_match is not None
        assert result.source_records
        assert result.field_comparisons

    def test_evidence_names_both_sources_titles_and_token_sets(self, fact_checker, monkeypatch):
        ev = _spliced_result(fact_checker, monkeypatch).chimeric_evidence
        assert ev is not None

        assert {ev.source_a, ev.source_b} == {"crossref", "dblp"}
        titles = {ev.source_a: ev.title_a, ev.source_b: ev.title_b}
        assert titles == {"crossref": REC_ATTENTION.title, "dblp": REC_SAE.title}

        shared = {ev.source_a: set(ev.shared_tokens_a), ev.source_b: set(ev.shared_tokens_b)}
        unique = {ev.source_a: set(ev.unique_tokens_a), ev.source_b: set(ev.unique_tokens_b)}
        assert shared["crossref"] == {"attention", "all", "you", "need"}
        assert shared["dblp"] == {"sparse", "autoencoder", "feature", "interpretability", "language", "systems"}
        # Nothing overlaps between the two papers, so unique == shared here.
        assert unique == shared
        assert ev.min_shared == 4
        assert ev.min_unique == 4
        assert (
            ev.entry_title
            == "attention is all you need sparse autoencoder feature interpretability in language systems"
        )

    def test_token_sets_are_sorted_for_stable_output(self, fact_checker, monkeypatch):
        ev = _spliced_result(fact_checker, monkeypatch).chimeric_evidence
        assert ev is not None
        for tokens in (ev.shared_tokens_a, ev.shared_tokens_b, ev.unique_tokens_a, ev.unique_tokens_b):
            assert list(tokens) == sorted(tokens)

    def test_best_match_is_the_higher_scoring_candidate(self, fact_checker, monkeypatch):
        # The entry's author is Vaswani, so the "Attention" record outscores
        # the SAE record on the author component and becomes record_a.
        result = _spliced_result(fact_checker, monkeypatch)
        ev = result.chimeric_evidence
        assert ev is not None

        assert ev.score_a >= ev.score_b
        assert result.best_match is ev.record_a
        assert result.best_match is REC_ATTENTION
        assert ev.source_a == "crossref"

    def test_best_match_follows_the_score_not_cascade_order(self, fact_checker, monkeypatch):
        # Same two records, but the entry now names the SAE paper's author, so
        # the second-in-cascade record scores higher and must be best_match.
        _stub_cascade(fact_checker, monkeypatch, [(REC_ATTENTION, "crossref"), (REC_SAE, "dblp")])
        result = fact_checker.check_entry({**SPLICED_ENTRY, "author": "Doe, Jane"})

        ev = result.chimeric_evidence
        assert ev is not None
        assert ev.score_a >= ev.score_b
        assert result.best_match is REC_SAE
        assert ev.source_a == "dblp"
        assert ev.source_b == "crossref"

    def test_source_records_hold_both_candidates_keyed_by_source(self, fact_checker, monkeypatch):
        result = _spliced_result(fact_checker, monkeypatch)
        assert result.source_records == {"crossref": REC_ATTENTION, "dblp": REC_SAE}

    def test_title_comparison_is_an_auditable_mismatch(self, fact_checker, monkeypatch):
        result = _spliced_result(fact_checker, monkeypatch)

        assert set(result.field_comparisons) == {"title"}
        comp = result.field_comparisons["title"]
        assert isinstance(comp, FieldComparison)
        assert comp.is_mismatch
        assert not comp.matches
        assert comp.entry_value == SPLICED_ENTRY["title"]
        assert comp.api_value == REC_ATTENTION.title
        assert 0.0 <= comp.similarity_score <= 1.0
        assert comp.note is not None
        for fragment in ("crossref", "dblp", REC_ATTENTION.title, REC_SAE.title, "autoencoder", "attention"):
            assert fragment in comp.note

    def test_error_string_cites_sources_and_token_counts(self, fact_checker, monkeypatch):
        result = _spliced_result(fact_checker, monkeypatch)

        msg = result.errors[-1]
        assert msg.startswith("Chimeric title detected")
        assert "crossref" in msg and "dblp" in msg
        assert "4 tokens shared with crossref" in msg
        assert "6 with dblp" in msg
        assert "4 of those appear only in the crossref title" in msg
        assert "6 only in the dblp title" in msg
        assert result.chimeric_evidence is not None
        assert msg == result.chimeric_evidence.summary()

    def test_confidence_is_the_margin_derived_value_not_a_constant(self, fact_checker, monkeypatch):
        result = _spliced_result(fact_checker, monkeypatch)
        ev = result.chimeric_evidence
        assert ev is not None

        assert result.overall_confidence == pytest.approx(ev.confidence)
        assert result.overall_confidence == pytest.approx(chimeric_confidence(4, 4))
        assert result.overall_confidence != 0.95
        assert CHIMERIC_CONFIDENCE_FLOOR <= result.overall_confidence < CHIMERIC_CONFIDENCE_CEILING
        # A problem-polarity verdict: P(valid) must sit well below neutral.
        assert result.p_valid < 0.2

    def test_verification_result_sees_the_match_and_the_mismatch(self, fact_checker, monkeypatch):
        vr = build_verification_result(_spliced_result(fact_checker, monkeypatch))

        assert vr.status == "hallucinated"
        assert vr.matched_metadata is not None
        assert vr.matched_metadata["title"] == REC_ATTENTION.title
        assert "title_mismatch" in vr.issues
        assert set(vr.sources_consulted) >= {"crossref", "dblp"}


# ------------- Confidence is a measurement -------------


class TestChimericConfidence:
    def test_threshold_exact_detection_reports_the_floor(self):
        assert chimeric_confidence(CHIMERIC_MIN_SHARED_TOKENS, CHIMERIC_MIN_UNIQUE_TOKENS) == pytest.approx(
            CHIMERIC_CONFIDENCE_FLOOR
        )
        assert chimeric_confidence(4, 3) == pytest.approx(0.80)

    def test_monotone_in_both_margins(self):
        for shared in range(5, 15):
            for unique in range(4, shared):
                conf = chimeric_confidence(shared, unique)
                assert conf > chimeric_confidence(shared - 1, unique)
                assert conf > chimeric_confidence(shared, unique - 1)
                assert conf > CHIMERIC_CONFIDENCE_FLOOR
        assert chimeric_confidence(9, 7) > chimeric_confidence(6, 5) > chimeric_confidence(4, 3)

    def test_saturates_below_the_ceiling(self):
        assert chimeric_confidence(9, 7) == pytest.approx(0.80 + 0.17 * 9 / 13)
        assert chimeric_confidence(100, 100) < CHIMERIC_CONFIDENCE_CEILING
        assert chimeric_confidence(10_000, 10_000) == pytest.approx(CHIMERIC_CONFIDENCE_CEILING, abs=1e-3)
        assert chimeric_confidence(10_000, 10_000) < CHIMERIC_CONFIDENCE_CEILING

    def test_negative_margin_is_clamped_to_the_floor(self):
        # The detector never produces this, but the formula must not go below floor.
        assert chimeric_confidence(0, 0) == pytest.approx(CHIMERIC_CONFIDENCE_FLOOR)

    def test_marginal_detection_scores_lower_than_overwhelming_one(self, fact_checker):
        # Exactly 4 shared / 3 unique per source: the weakest evidence that fires.
        marginal_entry = {"title": "alpha beta gamma delta epsilon zeta eta"}
        marginal = fact_checker._detect_chimeric_title(
            marginal_entry,
            _candidates(("crossref", "alpha beta gamma delta", 0.6), ("dblp", "alpha epsilon zeta eta", 0.5)),
        )
        # 9 shared / 7 unique per source: two long titles sharing two tokens.
        big_entry = {
            "title": (
                "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
            )
        }
        big = fact_checker._detect_chimeric_title(
            big_entry,
            _candidates(
                ("crossref", "alpha beta gamma delta epsilon zeta eta theta iota", 0.6),
                ("dblp", "alpha beta kappa lambda mu nu xi omicron pi", 0.5),
            ),
        )

        assert marginal is not None and big is not None
        assert (marginal.min_shared, marginal.min_unique) == (4, 3)
        assert (big.min_shared, big.min_unique) == (9, 7)
        assert marginal.confidence == pytest.approx(CHIMERIC_CONFIDENCE_FLOOR)
        assert big.confidence > marginal.confidence
        assert big.confidence == pytest.approx(chimeric_confidence(9, 7))

    def test_strongest_pair_wins_when_several_qualify(self, fact_checker):
        entry = {"title": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"}
        ev = fact_checker._detect_chimeric_title(
            entry,
            _candidates(
                ("crossref", "alpha beta gamma delta", 0.5),  # 4 shared
                ("dblp", "alpha epsilon zeta eta", 0.5),  # 4 shared; vs crossref: 3/3 unique
                ("openalex", "theta iota kappa lambda mu nu", 0.5),  # 6 shared; vs either: 6/4 unique
            ),
        )
        assert ev is not None
        assert "openalex" in (ev.source_a, ev.source_b)
        assert ev.min_shared == 4
        assert ev.min_unique == 4


# ------------- Reports carry the evidence, additively -------------


def _verified_result() -> FactCheckResult:
    return FactCheckResult(
        entry_key="ok2020",
        entry_type="article",
        status=FactCheckStatus.VERIFIED,
        overall_confidence=0.95,
        field_comparisons={"title": FieldComparison("title", "T", "T", 1.0, True)},
        best_match=REC_ATTENTION,
        api_sources_queried=["crossref"],
        api_sources_with_hits=["crossref"],
        errors=[],
    )


EVIDENCE_KEYS = {
    "entry_title",
    "source_a",
    "source_b",
    "title_a",
    "title_b",
    "doi_a",
    "doi_b",
    "score_a",
    "score_b",
    "shared_tokens_a",
    "shared_tokens_b",
    "unique_tokens_a",
    "unique_tokens_b",
    "min_shared",
    "min_unique",
    "thresholds",
    "confidence",
}


class TestReportSerialization:
    def test_json_report_includes_evidence_and_keeps_existing_keys(self, fact_checker, processor, monkeypatch):
        chimeric = _spliced_result(fact_checker, monkeypatch)
        report = processor.generate_json_report([_verified_result(), chimeric])
        json.dumps(report)  # must be serializable end to end

        ok_entry, bad_entry = report["entries"]
        assert "chimeric_evidence" not in ok_entry
        # Every pre-existing key survives on the chimeric entry.
        assert set(ok_entry) <= set(bad_entry)

        ev = bad_entry["chimeric_evidence"]
        assert set(ev) == EVIDENCE_KEYS
        assert {ev["source_a"], ev["source_b"]} == {"crossref", "dblp"}
        assert {ev["title_a"], ev["title_b"]} == {REC_ATTENTION.title, REC_SAE.title}
        assert {ev["doi_a"], ev["doi_b"]} == {REC_ATTENTION.doi, REC_SAE.doi}
        assert ev["thresholds"] == {"shared": CHIMERIC_MIN_SHARED_TOKENS, "unique": CHIMERIC_MIN_UNIQUE_TOKENS}
        assert ev["confidence"] == pytest.approx(bad_entry["confidence"])
        assert bad_entry["best_match"]["doi"] == REC_ATTENTION.doi
        assert bad_entry["field_comparisons"]["title"]["matches"] is False
        assert "crossref" in bad_entry["field_comparisons"]["title"]["note"]

    def test_jsonl_includes_evidence_and_keeps_existing_keys(self, fact_checker, processor, monkeypatch):
        chimeric = _spliced_result(fact_checker, monkeypatch)
        ok_line, bad_line = (json.loads(line) for line in processor.generate_jsonl([_verified_result(), chimeric]))

        assert "chimeric_evidence" not in ok_line
        assert set(ok_line) <= set(bad_line)
        assert set(bad_line) - set(ok_line) == {"chimeric_evidence"}
        assert set(bad_line["chimeric_evidence"]) == EVIDENCE_KEYS
        assert bad_line["status"] == "hallucinated"
        assert bad_line["mismatched_fields"] == ["title"]
        assert bad_line["unconfirmed_fields"] == []
        assert bad_line["confidence"] == pytest.approx(chimeric.overall_confidence)

    def test_incremental_jsonl_from_process_entries_matches(self, fact_checker, processor, monkeypatch, tmp_path):
        _stub_cascade(fact_checker, monkeypatch, [(REC_ATTENTION, "crossref"), (REC_SAE, "dblp")])
        out = tmp_path / "out.jsonl"
        results = processor.process_entries([dict(SPLICED_ENTRY)], jsonl_path=str(out))

        assert len(results) == 1
        (line,) = (json.loads(ln) for ln in out.read_text().strip().splitlines())
        assert line["status"] == "hallucinated"
        assert set(line["chimeric_evidence"]) == EVIDENCE_KEYS
        assert line["chimeric_evidence"] == results[0].chimeric_evidence.to_dict()

    def test_to_dict_round_trips_through_json(self, fact_checker, monkeypatch):
        ev = _spliced_result(fact_checker, monkeypatch).chimeric_evidence
        assert isinstance(ev, ChimericEvidence)
        payload = json.loads(json.dumps(ev.to_dict()))
        assert payload["shared_tokens_a"] == list(ev.shared_tokens_a)
        assert payload["min_shared"] == ev.min_shared
        assert payload["confidence"] == pytest.approx(ev.confidence)


# ------------- The rule itself is unchanged -------------


class TestNegativeCasesStillPass:
    def test_single_source_is_never_chimeric(self, fact_checker):
        ev = fact_checker._detect_chimeric_title(
            dict(SPLICED_ENTRY), _candidates(("crossref", REC_ATTENTION.title, 0.6), ("crossref", REC_SAE.title, 0.5))
        )
        assert ev is None

    def test_fewer_than_two_candidates_is_never_chimeric(self, fact_checker):
        assert fact_checker._detect_chimeric_title(dict(SPLICED_ENTRY), []) is None
        assert (
            fact_checker._detect_chimeric_title(dict(SPLICED_ENTRY), _candidates(("crossref", REC_SAE.title, 0.9)))
            is None
        )

    def test_preprint_and_published_title_variants_are_not_chimeric(self, fact_checker):
        # Two sources, same paper, slightly different titles: the shared token
        # sets nearly coincide, so neither side contributes 3 unique tokens.
        entry = {"title": "Sparse Autoencoder Feature Interpretability in Language Systems"}
        ev = fact_checker._detect_chimeric_title(
            entry,
            _candidates(
                ("crossref", "Sparse Autoencoder Feature Interpretability in Language Systems", 0.95),
                ("arxiv", "Sparse Autoencoder Feature Interpretability for Large Language Systems", 0.9),
            ),
        )
        assert ev is None

    def test_below_shared_threshold_is_not_chimeric(self, fact_checker):
        # Each source shares only 3 tokens with the entry: below the >= 4 rule.
        entry = {"title": "alpha beta gamma epsilon zeta eta"}
        ev = fact_checker._detect_chimeric_title(
            entry, _candidates(("crossref", "alpha beta gamma", 0.6), ("dblp", "epsilon zeta eta", 0.6))
        )
        assert ev is None

    def test_below_unique_threshold_is_not_chimeric(self, fact_checker):
        # Both share 4 tokens, but only 2 differ per side: below the >= 3 rule.
        entry = {"title": "alpha beta gamma delta epsilon zeta"}
        ev = fact_checker._detect_chimeric_title(
            entry, _candidates(("crossref", "alpha beta gamma delta", 0.6), ("dblp", "alpha beta epsilon zeta", 0.6))
        )
        assert ev is None

    def test_truthiness_matches_the_old_bool_contract(self, fact_checker):
        assert not fact_checker._detect_chimeric_title(dict(SPLICED_ENTRY), [])
        assert fact_checker._detect_chimeric_title(
            dict(SPLICED_ENTRY), _candidates(("crossref", REC_ATTENTION.title, 0.6), ("dblp", REC_SAE.title, 0.5))
        )

    def test_non_chimeric_result_has_no_evidence_field(self, fact_checker, monkeypatch):
        _stub_cascade(fact_checker, monkeypatch, [(REC_SAE, "crossref")])
        result = fact_checker.check_entry(
            {"ID": "real", "ENTRYTYPE": "article", "title": REC_SAE.title, "author": "Doe, Jane", "year": "2024"}
        )
        assert result.status is not FactCheckStatus.HALLUCINATED
        assert result.chimeric_evidence is None
