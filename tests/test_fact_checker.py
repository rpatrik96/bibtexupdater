"""Tests for reference_fact_checker module."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from bibtex_updater.fact_checker import (
    CrossrefClient,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    FactCheckProcessor,
    FactCheckResult,
    FactCheckStatus,
    FieldComparison,
    SemanticScholarClient,
)
from bibtex_updater.matching import MatchOutcome
from bibtex_updater.utils import PublishedRecord

# ------------- Fixtures -------------


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test_fact_checker")


@pytest.fixture
def fake_http():
    """Create a mock HTTP client."""
    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return mock


@pytest.fixture
def fake_crossref(fake_http):
    """Create CrossrefClient with mock HTTP."""
    client = CrossrefClient(fake_http)
    return client


@pytest.fixture
def fake_dblp(fake_http):
    """Create DBLPClient with mock HTTP."""
    return DBLPClient(fake_http)


@pytest.fixture
def fake_s2(fake_http):
    """Create SemanticScholarClient with mock HTTP."""
    return SemanticScholarClient(fake_http)


@pytest.fixture
def fact_checker_config():
    """Default FactCheckerConfig."""
    return FactCheckerConfig()


@pytest.fixture
def fact_checker(fake_crossref, fake_dblp, fake_s2, fact_checker_config, logger):
    """FactChecker with mocked API clients."""
    return FactChecker(fake_crossref, fake_dblp, fake_s2, fact_checker_config, logger)


@pytest.fixture
def processor(fact_checker, logger):
    """FactCheckProcessor instance."""
    return FactCheckProcessor(fact_checker, logger)


@pytest.fixture
def sample_entry():
    """A sample BibTeX entry."""
    return {
        "ID": "smith2020",
        "ENTRYTYPE": "article",
        "title": "Deep Learning for Natural Language Processing",
        "author": "Smith, John and Doe, Jane",
        "journal": "Journal of Machine Learning",
        "year": "2020",
    }


@pytest.fixture
def sample_published_record():
    """A sample PublishedRecord matching sample_entry."""
    return PublishedRecord(
        doi="10.1234/jml.2020.001",
        title="Deep Learning for Natural Language Processing",
        authors=[
            {"given": "John", "family": "Smith"},
            {"given": "Jane", "family": "Doe"},
        ],
        journal="Journal of Machine Learning",
        year=2020,
    )


# ------------- FactCheckerConfig Tests -------------


class TestFactCheckerConfig:
    """Tests for FactCheckerConfig."""

    def test_default_values(self):
        cfg = FactCheckerConfig()
        assert cfg.title_threshold == 0.90
        assert cfg.author_threshold == 0.80
        assert cfg.year_tolerance == 1
        assert cfg.venue_threshold == 0.70
        assert cfg.hallucination_max_score == 0.50

    def test_custom_values(self):
        cfg = FactCheckerConfig(title_threshold=0.85, year_tolerance=2, hallucination_max_score=0.40)
        assert cfg.title_threshold == 0.85
        assert cfg.year_tolerance == 2
        assert cfg.hallucination_max_score == 0.40


# ------------- FieldComparison Tests -------------


class TestFieldComparison:
    """Tests for FieldComparison dataclass."""

    def test_matching_field(self):
        comp = FieldComparison(
            field_name="title",
            entry_value="Deep Learning",
            api_value="Deep Learning",
            similarity_score=1.0,
            matches=True,
        )
        assert comp.matches is True
        assert comp.similarity_score == 1.0

    def test_mismatched_field(self):
        comp = FieldComparison(
            field_name="title",
            entry_value="Original Title",
            api_value="Different Title",
            similarity_score=0.5,
            matches=False,
        )
        assert comp.matches is False
        assert comp.similarity_score == 0.5


# ------------- FactCheckResult Tests -------------


class TestFactCheckResult:
    """Tests for FactCheckResult dataclass."""

    def test_verified_result(self, sample_published_record):
        result = FactCheckResult(
            entry_key="test2020",
            entry_type="article",
            status=FactCheckStatus.VERIFIED,
            overall_confidence=0.95,
            field_comparisons={},
            best_match=sample_published_record,
            api_sources_queried=["crossref"],
            api_sources_with_hits=["crossref"],
            errors=[],
        )
        assert result.status == FactCheckStatus.VERIFIED
        assert result.overall_confidence == 0.95
        assert result.best_match is not None

    def test_not_found_result(self):
        result = FactCheckResult(
            entry_key="unknown2020",
            entry_type="article",
            status=FactCheckStatus.NOT_FOUND,
            overall_confidence=0.0,
            field_comparisons={},
            best_match=None,
            api_sources_queried=["crossref", "dblp"],
            api_sources_with_hits=[],
            errors=[],
        )
        assert result.status == FactCheckStatus.NOT_FOUND
        assert result.best_match is None


# ------------- FactChecker Tests -------------


class TestFactCheckerScoring:
    """Tests for FactChecker scoring methods."""

    def test_score_perfect_match(self, fact_checker, sample_published_record):
        title_norm = "deep learning for natural language processing"
        authors_ref = ["smith", "doe"]
        score = fact_checker._score_candidate(title_norm, authors_ref, sample_published_record)
        assert score > 0.90

    def test_score_title_mismatch(self, fact_checker):
        rec = PublishedRecord(
            doi="10.1234/test",
            title="Completely Different Topic Here",
            authors=[{"given": "John", "family": "Smith"}],
        )
        score = fact_checker._score_candidate("machine learning basics", ["smith"], rec)
        assert score < 0.70


class TestFactCheckerFieldComparison:
    """Tests for FactChecker field comparison."""

    def test_compare_identical_fields(self, fact_checker, sample_entry, sample_published_record):
        comparisons = fact_checker._compare_all_fields(sample_entry, sample_published_record)

        assert "title" in comparisons
        assert comparisons["title"].matches is True

        assert "author" in comparisons
        assert comparisons["author"].matches is True

        assert "year" in comparisons
        assert comparisons["year"].matches is True

    def test_compare_year_within_tolerance(self, fact_checker):
        entry = {"title": "Test", "year": "2020", "author": "Author"}
        record = PublishedRecord(doi="", title="Test", year=2021)
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["year"].matches is True

    def test_compare_year_outside_tolerance(self, fact_checker):
        entry = {"title": "Test", "year": "2018", "author": "Author"}
        record = PublishedRecord(doi="", title="Test", year=2021)
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["year"].matches is False

    def test_compare_no_venue_claim_is_vacuously_confirmed(self, fact_checker):
        """Entry makes NO venue claim -> nothing to confirm -> vacuous MATCH.

        A preprint @misc/@article with no journal/booktitle must not be blocked
        from VERIFIED just because it omits a venue: there is no claim to verify.
        """
        from bibtex_updater.matching import MatchOutcome

        entry = {"title": "Test", "author": "Author"}  # No journal/booktitle
        record = PublishedRecord(doi="", title="Test", journal="Some Journal")
        comparisons = fact_checker._compare_all_fields(entry, record)
        venue = comparisons["venue"]
        assert venue.outcome is MatchOutcome.MATCH
        assert venue.matches is True

    def test_compare_claimed_venue_vs_preprint_record_is_non_comparable(self, fact_checker):
        """Entry CLAIMS a published venue but the record is preprint-only ->
        NON_COMPARABLE: not a mismatch, but not a positive confirmation either."""
        from bibtex_updater.matching import MatchOutcome

        entry = {"title": "Test", "author": "Smith, John", "booktitle": "NeurIPS"}
        record = PublishedRecord(
            doi="",
            title="Test",
            authors=[{"given": "John", "family": "Smith"}],
            journal="arXiv preprint arXiv:2010.11929",
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        venue = comparisons["venue"]
        assert venue.outcome is MatchOutcome.NON_COMPARABLE
        assert venue.is_mismatch is False
        assert venue.matches is False  # cannot confirm the claimed published venue

    def test_compare_author_subset_without_sentinel_is_partial(self, fact_checker):
        """FIX C / positive-confirmation: entry lists first 3 of 8 authors with no
        elision sentinel -> consistent but INCOMPLETE -> PARTIAL, not a full match.

        The legacy asymmetric comparison scored ~0.375 (false mismatch); the prior
        softening folded this into a full match. A silent leading subset is now a
        PARTIAL confirmation: not a mismatch, but not a positive confirmation either.
        """
        from bibtex_updater.matching import MatchOutcome

        entry = {
            "title": "An Image Is Worth 16x16 Words",
            "author": "Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander",
        }
        record = PublishedRecord(
            doi="10.48550/arXiv.2010.11929",
            title="An Image Is Worth 16x16 Words",
            authors=[
                {"given": "Alexey", "family": "Dosovitskiy"},
                {"given": "Lucas", "family": "Beyer"},
                {"given": "Alexander", "family": "Kolesnikov"},
                {"given": "Dirk", "family": "Weissenborn"},
                {"given": "Xiaohua", "family": "Zhai"},
                {"given": "Thomas", "family": "Unterthiner"},
                {"given": "Mostafa", "family": "Dehghani"},
                {"given": "Matthias", "family": "Minderer"},
            ],
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        author = comparisons["author"]
        assert author.outcome is MatchOutcome.PARTIAL
        assert author.is_mismatch is False
        assert author.matches is False  # incomplete is NOT a full confirmation

    def test_compare_author_and_others_matches(self, fact_checker):
        """FIX C: 'and others' must not introduce a phantom mismatch author."""
        entry = {
            "title": "Test",
            "author": "Smith, John and Doe, Jane and others",
        }
        record = PublishedRecord(
            doi="",
            title="Test",
            authors=[
                {"given": "John", "family": "Smith"},
                {"given": "Jane", "family": "Doe"},
                {"given": "Alan", "family": "Turing"},
            ],
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["author"].matches is True

    def test_compare_different_first_author_still_fails(self, fact_checker):
        """FIX C: a genuinely wrong lead author must still be flagged."""
        entry = {"title": "Test", "author": "Wrong, Person and Doe, Jane"}
        record = PublishedRecord(
            doi="",
            title="Test",
            authors=[
                {"given": "John", "family": "Smith"},
                {"given": "Jane", "family": "Doe"},
            ],
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["author"].matches is False


class TestFactCheckerStatusDetermination:
    """Tests for FactChecker status determination."""

    def test_verified_all_match(self, fact_checker):
        comparisons = {
            "title": FieldComparison("title", "A", "A", 1.0, True),
            "author": FieldComparison("author", "B", "B", 1.0, True),
            "year": FieldComparison("year", "2021", "2021", 1.0, True),
            "venue": FieldComparison("venue", "J", "J", 1.0, True),
        }
        status = fact_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status == FactCheckStatus.VERIFIED

    def test_low_score_abstains_not_hallucinated(self, fact_checker):
        # Fix B: a weak best match means the title search returned an unrelated
        # paper -- the tool could not verify, which is NOT positive evidence of
        # fabrication. It must ABSTAIN (NOT_FOUND), not assert HALLUCINATED.
        comparisons = {"title": FieldComparison("title", "A", "B", 0.3, False)}
        status = fact_checker._determine_status(0.35, comparisons, [])
        assert status == FactCheckStatus.NOT_FOUND

    def test_wrong_paper_signature_abstains(self, fact_checker):
        # Above the abstention score threshold, but the title is essentially
        # unrelated and neither title nor author corroborate -> still abstain.
        comparisons = {
            "title": FieldComparison("title", "A", "Z", 0.10, False),
            "author": FieldComparison("author", "X", "Q", 0.0, False),
            "year": FieldComparison("year", "2021", "2021", 1.0, True),
            "venue": FieldComparison("venue", "J", "J", 1.0, True),
        }
        status = fact_checker._determine_status(0.55, comparisons, ["crossref"])
        assert status == FactCheckStatus.NOT_FOUND

    def test_title_mismatch_only(self, fact_checker):
        comparisons = {
            "title": FieldComparison("title", "A", "B", 0.5, False),
            "author": FieldComparison("author", "X", "X", 1.0, True),
            "year": FieldComparison("year", "2021", "2021", 1.0, True),
            "venue": FieldComparison("venue", "J", "J", 1.0, True),
        }
        status = fact_checker._determine_status(0.75, comparisons, ["crossref"])
        assert status == FactCheckStatus.TITLE_MISMATCH

    def test_author_mismatch_only(self, fact_checker):
        comparisons = {
            "title": FieldComparison("title", "A", "A", 1.0, True),
            "author": FieldComparison("author", "X", "Y", 0.5, False),
            "year": FieldComparison("year", "2021", "2021", 1.0, True),
            "venue": FieldComparison("venue", "J", "J", 1.0, True),
        }
        status = fact_checker._determine_status(0.80, comparisons, ["crossref"])
        assert status == FactCheckStatus.AUTHOR_MISMATCH

    def test_year_mismatch_only(self, fact_checker):
        comparisons = {
            "title": FieldComparison("title", "A", "A", 1.0, True),
            "author": FieldComparison("author", "X", "X", 1.0, True),
            "year": FieldComparison("year", "2018", "2021", 0.0, False),
            "venue": FieldComparison("venue", "J", "J", 1.0, True),
        }
        status = fact_checker._determine_status(0.85, comparisons, ["crossref"])
        assert status == FactCheckStatus.YEAR_MISMATCH

    def test_partial_match_multiple_mismatches(self, fact_checker):
        comparisons = {
            "title": FieldComparison("title", "A", "B", 0.6, False),
            "author": FieldComparison("author", "X", "Y", 0.5, False),
            "year": FieldComparison("year", "2021", "2021", 1.0, True),
            "venue": FieldComparison("venue", "J", "J", 1.0, True),
        }
        status = fact_checker._determine_status(0.70, comparisons, ["crossref"])
        assert status == FactCheckStatus.PARTIAL_MATCH


class TestPositiveConfirmationGate:
    """VERIFIED requires positive confirmation of EVERY claimed field.

    A field that is neither a confirmed MATCH nor a real MISMATCH
    (NON_COMPARABLE venue / PARTIAL author) must route to UNCONFIRMED -- a
    could-not-verify abstention -- not VERIFIED and not a *_MISMATCH.
    """

    def _confirmed(self, name):
        return FieldComparison(name, "x", "x", 1.0, True, outcome=MatchOutcome.MATCH)

    def test_non_comparable_venue_routes_to_unconfirmed(self, fact_checker):
        comparisons = {
            "title": self._confirmed("title"),
            "author": self._confirmed("author"),
            "year": self._confirmed("year"),
            "venue": FieldComparison("venue", "NeurIPS", "", 1.0, False, outcome=MatchOutcome.NON_COMPARABLE),
        }
        status = fact_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status == FactCheckStatus.UNCONFIRMED

    def test_partial_author_routes_to_unconfirmed(self, fact_checker):
        comparisons = {
            "title": self._confirmed("title"),
            "author": FieldComparison("author", "a,b", "a,b,c,d", 1.0, False, outcome=MatchOutcome.PARTIAL),
            "year": self._confirmed("year"),
            "venue": self._confirmed("venue"),
        }
        status = fact_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status == FactCheckStatus.UNCONFIRMED

    def test_all_confirmed_is_verified(self, fact_checker):
        comparisons = {
            "title": self._confirmed("title"),
            "author": self._confirmed("author"),
            "year": self._confirmed("year"),
            "venue": self._confirmed("venue"),
        }
        status = fact_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status == FactCheckStatus.VERIFIED

    def test_real_venue_mismatch_beats_unconfirmed(self, fact_checker):
        """A real MISMATCH is positive evidence -> PROBLEMATIC, taking priority
        over a co-occurring non-confirming field."""
        comparisons = {
            "title": self._confirmed("title"),
            "author": self._confirmed("author"),
            "year": self._confirmed("year"),
            "venue": FieldComparison("venue", "NeurIPS", "ICML", 0.0, False, outcome=MatchOutcome.MISMATCH),
        }
        status = fact_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status == FactCheckStatus.VENUE_MISMATCH

    def test_unconfirmed_is_an_abstention_not_a_problem(self):
        """UNCONFIRMED belongs in the could-not-verify (abstained) bucket."""
        from bibtex_updater.fact_checker import _is_abstained_status

        assert _is_abstained_status(FactCheckStatus.UNCONFIRMED) is True


class TestDetectionRatePreservation:
    """Fix B guard: positive-evidence hallucination signals must STILL yield
    HALLUCINATED. These signals fire *before* ``_determine_status`` (in
    ``check_entry``), so the new abstention rule cannot suppress them.
    """

    def test_future_date_still_flagged(self, fact_checker):
        # _validate_year runs before any title search / score gate.
        entry = {
            "ID": "future",
            "ENTRYTYPE": "article",
            "title": "Some Real Title",
            "author": "Smith, John",
            "year": "2099",
        }
        result = fact_checker.check_entry(entry)
        assert result.status == FactCheckStatus.FUTURE_DATE
        assert result.status != FactCheckStatus.NOT_FOUND

    def test_invalid_year_still_flagged(self, fact_checker):
        entry = {
            "ID": "badyear",
            "ENTRYTYPE": "article",
            "title": "Some Real Title",
            "author": "Smith, John",
            "year": "1500",
        }
        result = fact_checker.check_entry(entry)
        assert result.status == FactCheckStatus.INVALID_YEAR
        assert result.status != FactCheckStatus.NOT_FOUND

    def test_fabricated_doi_still_flagged(self, fact_checker):
        # Pre-validated DOI map marks the DOI invalid (404/410). _validate_doi
        # returns DOI_NOT_FOUND before the title search runs.
        entry = {
            "ID": "fakedoi",
            "ENTRYTYPE": "article",
            "title": "Some Real Title",
            "author": "Smith, John",
            "year": "2020",
            "doi": "10.9999/totally.made.up",
        }
        result = fact_checker.check_entry(entry, pre_validated_dois={"fakedoi": False})
        assert result.status == FactCheckStatus.DOI_NOT_FOUND
        assert result.status != FactCheckStatus.NOT_FOUND

    def test_chimeric_title_still_flagged(self, fact_checker, monkeypatch):
        # Chimeric detection fires on positive cross-source evidence BEFORE the
        # candidate sort and score gate. Force the cascade to return two records
        # from two sources whose titles each contribute distinct token sets that
        # the entry title borrows from.
        entry = {
            "ID": "chimera",
            "ENTRYTYPE": "article",
            # Borrows "attention transformers sequence modeling" from one paper
            # and "graph convolutional molecular property prediction" from
            # another (>= 4 shared tokens per source, so the chimeric detector
            # fires -- which is the point of this DR-preservation guard).
            "title": "Attention Transformers Sequence Modeling Graph Convolutional Molecular Property Prediction",
            "author": "Smith, John",
            "year": "2020",
        }
        rec_a = PublishedRecord(
            doi="10.1/a",
            title="Attention Transformers Sequence Modeling Translation",
            authors=[{"given": "John", "family": "Smith"}],
            journal="JMLR",
            year=2020,
        )
        rec_b = PublishedRecord(
            doi="10.1/b",
            title="Graph Convolutional Molecular Property Prediction Networks",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="NeurIPS",
            year=2020,
        )

        def fake_cascade(entry, query, sq, sh, errors):
            sq.extend(["crossref", "dblp"])
            sh.extend(["crossref", "dblp"])
            return [
                (fact_checker._score_candidate(query, ["smith"], rec_a), rec_a, "crossref"),
                (fact_checker._score_candidate(query, ["smith"], rec_b), rec_b, "dblp"),
            ]

        monkeypatch.setattr(fact_checker, "_query_cascade", fake_cascade)
        monkeypatch.setattr(fact_checker, "_query_arxiv_by_id", lambda *a, **k: [])
        result = fact_checker.check_entry(entry)
        assert result.status == FactCheckStatus.HALLUCINATED
        assert result.status != FactCheckStatus.NOT_FOUND


class TestFactCheckerCheckEntry:
    """Tests for FactChecker.check_entry method."""

    def test_entry_without_title(self, fact_checker):
        entry = {"ID": "notitle", "ENTRYTYPE": "article", "author": "Smith"}
        result = fact_checker.check_entry(entry)
        assert result.status == FactCheckStatus.API_ERROR
        assert "No title" in result.errors[0]

    def test_entry_not_found(self, fact_checker, sample_entry):
        # All API clients return empty results by default. The cascade now
        # queries four sources: crossref -> openalex -> dblp -> semanticscholar.
        result = fact_checker.check_entry(sample_entry)
        assert result.status == FactCheckStatus.NOT_FOUND
        assert result.api_sources_queried == ["crossref", "openalex", "dblp", "semanticscholar"]

    def test_weak_unrelated_match_abstains(self, fact_checker, monkeypatch):
        # Fix B: the search returns an UNRELATED paper (low best_score). This is
        # a failed lookup, not positive evidence of fabrication -> NOT_FOUND,
        # never HALLUCINATED.
        entry = {
            "ID": "weak",
            "ENTRYTYPE": "article",
            "title": "A Very Specific Real Paper Title That Was Not Indexed",
            "author": "Smith, John",
            "year": "2020",
        }
        unrelated = PublishedRecord(
            doi="10.1/unrelated",
            title="Completely Different Topic About Quantum Chromodynamics",
            authors=[{"given": "Zed", "family": "Other"}],
            journal="Physics Today",
            year=2019,
        )

        def fake_cascade(entry, query, sq, sh, errors):
            sq.append("crossref")
            sh.append("crossref")
            return [(fact_checker._score_candidate(query, ["smith"], unrelated), unrelated, "crossref")]

        monkeypatch.setattr(fact_checker, "_query_cascade", fake_cascade)
        monkeypatch.setattr(fact_checker, "_query_arxiv_by_id", lambda *a, **k: [])
        result = fact_checker.check_entry(entry)
        assert result.status == FactCheckStatus.NOT_FOUND
        assert result.status != FactCheckStatus.HALLUCINATED


# ------------- FactCheckProcessor Tests -------------


class TestFactCheckProcessor:
    """Tests for FactCheckProcessor."""

    def test_generate_summary_empty(self, processor):
        summary = processor.generate_summary([])
        assert summary["total"] == 0
        assert summary["verified_rate"] == 0

    def test_generate_summary_with_results(self, processor, sample_published_record):
        results = [
            FactCheckResult(
                "a",
                "article",
                FactCheckStatus.VERIFIED,
                0.95,
                {},
                sample_published_record,
                ["crossref"],
                ["crossref"],
                [],
            ),
            FactCheckResult(
                "b",
                "article",
                FactCheckStatus.NOT_FOUND,
                0.0,
                {},
                None,
                ["crossref"],
                [],
                [],
            ),
            FactCheckResult(
                "c",
                "article",
                FactCheckStatus.HALLUCINATED,
                0.3,
                {},
                None,
                ["crossref"],
                [],
                [],
            ),
        ]
        summary = processor.generate_summary(results)

        assert summary["total"] == 3
        assert summary["status_counts"]["verified"] == 1
        assert summary["status_counts"]["not_found"] == 1
        assert summary["status_counts"]["hallucinated"] == 1
        # Fix B: not_found is now an ABSTENTION ("could not verify"), reported
        # separately from PROBLEMATIC (positive evidence). Only the hallucinated
        # entry is problematic; the not_found entry is abstained.
        assert summary["problematic_count"] == 1
        assert summary["abstained_count"] == 1
        assert summary["verified_count"] == 1

    def test_summary_buckets_partition_unconfirmed_and_partial(self, processor, sample_published_record):
        """The three buckets must partition all entries: UNCONFIRMED -> could-not-
        verify (abstained); PARTIAL_MATCH -> problematic; VERIFIED -> verified."""

        def _r(key, status):
            return FactCheckResult(key, "article", status, 0.7, {}, None, ["crossref"], ["crossref"], [])

        results = [
            _r("v", FactCheckStatus.VERIFIED),
            _r("u", FactCheckStatus.UNCONFIRMED),
            _r("p", FactCheckStatus.PARTIAL_MATCH),
            _r("n", FactCheckStatus.NOT_FOUND),
        ]
        summary = processor.generate_summary(results)
        assert summary["verified_count"] == 1
        # could-not-verify = unconfirmed + not_found
        assert summary["abstained_count"] == 2
        # problematic = partial_match (now included so buckets fully partition)
        assert summary["problematic_count"] == 1
        # All four entries are accounted for across the three buckets.
        assert (
            summary["verified_count"] + summary["abstained_count"] + summary["problematic_count"] == summary["total"]
        )

    def test_generate_json_report(self, processor, sample_published_record):
        results = [
            FactCheckResult(
                "test2020",
                "article",
                FactCheckStatus.VERIFIED,
                0.95,
                {
                    "title": FieldComparison("title", "T", "T", 1.0, True),
                },
                sample_published_record,
                ["crossref"],
                ["crossref"],
                [],
            ),
        ]
        report = processor.generate_json_report(results)

        assert "summary" in report
        assert "entries" in report
        assert len(report["entries"]) == 1
        assert report["entries"][0]["key"] == "test2020"
        assert report["entries"][0]["status"] == "verified"
        assert "timestamp" in report["summary"]

    def test_generate_jsonl(self, processor, sample_published_record):
        results = [
            FactCheckResult(
                "test2020",
                "article",
                FactCheckStatus.VERIFIED,
                0.95,
                {"title": FieldComparison("title", "A", "A", 1.0, True)},
                sample_published_record,
                ["crossref"],
                ["crossref"],
                [],
            ),
            FactCheckResult(
                "missing2020",
                "article",
                FactCheckStatus.TITLE_MISMATCH,
                0.7,
                {"title": FieldComparison("title", "A", "B", 0.5, False)},
                None,
                ["crossref"],
                ["crossref"],
                [],
            ),
        ]
        lines = processor.generate_jsonl(results)

        assert len(lines) == 2
        assert '"status": "verified"' in lines[0]
        assert '"status": "title_mismatch"' in lines[1]
        assert '"mismatched_fields": ["title"]' in lines[1]

    def test_field_mismatch_counts(self, processor):
        results = [
            FactCheckResult(
                "a",
                "article",
                FactCheckStatus.TITLE_MISMATCH,
                0.7,
                {
                    "title": FieldComparison("title", "A", "B", 0.5, False),
                    "author": FieldComparison("author", "X", "X", 1.0, True),
                },
                None,
                [],
                [],
                [],
            ),
            FactCheckResult(
                "b",
                "article",
                FactCheckStatus.PARTIAL_MATCH,
                0.6,
                {
                    "title": FieldComparison("title", "A", "B", 0.5, False),
                    "year": FieldComparison("year", "2018", "2021", 0.0, False),
                },
                None,
                [],
                [],
                [],
            ),
        ]
        summary = processor.generate_summary(results)
        assert summary["field_mismatch_counts"]["title"] == 2
        assert summary["field_mismatch_counts"]["year"] == 1


# ------------- API Client Tests -------------


class TestCrossrefClient:
    """Tests for CrossrefClient."""

    def test_search_returns_empty_on_error(self, fake_http):
        fake_http._request.side_effect = Exception("Network error")
        client = CrossrefClient(fake_http)
        results = client.search("test query")
        assert results == []

    def test_search_returns_empty_on_non_200(self, fake_http):
        fake_http._request.return_value = MagicMock(status_code=500)
        client = CrossrefClient(fake_http)
        results = client.search("test query")
        assert results == []


class TestDBLPClient:
    """Tests for DBLPClient."""

    def test_search_returns_empty_on_error(self, fake_http):
        fake_http._request.side_effect = Exception("Network error")
        client = DBLPClient(fake_http)
        results = client.search("test query")
        assert results == []

    def test_search_handles_single_hit_as_dict(self, fake_http):
        fake_http._request.return_value = MagicMock(
            status_code=200, json=lambda: {"result": {"hits": {"hit": {"info": {"title": "Single"}}}}}
        )
        client = DBLPClient(fake_http)
        results = client.search("test")
        assert len(results) == 1


class TestSemanticScholarClient:
    """Tests for SemanticScholarClient."""

    def test_search_returns_empty_on_error(self, fake_http):
        fake_http._request.side_effect = Exception("Network error")
        client = SemanticScholarClient(fake_http)
        results = client.search("test query")
        assert results == []

    def test_search_returns_empty_on_non_200(self, fake_http):
        fake_http._request.return_value = MagicMock(status_code=429)
        client = SemanticScholarClient(fake_http)
        results = client.search("test query")
        assert results == []


# ------------- Entry Classification Tests -------------


class TestEntryClassifier:
    """Tests for EntryClassifier."""

    @pytest.fixture
    def classifier(self):
        from bibtex_updater.fact_checker import EntryClassifier

        return EntryClassifier()

    def test_classify_book(self, classifier):
        from bibtex_updater.fact_checker import EntryCategory

        entry = {
            "ENTRYTYPE": "book",
            "ID": "test2023",
            "title": "Test Book",
            "author": "Smith, John",
            "publisher": "Test Publisher",
        }
        result = classifier.classify(entry)
        assert result.category == EntryCategory.BOOK
        assert "book" in result.reason.lower()

    def test_classify_inbook(self, classifier):
        from bibtex_updater.fact_checker import EntryCategory

        entry = {
            "ENTRYTYPE": "inbook",
            "ID": "test2023",
            "title": "Test Chapter",
            "author": "Smith, John",
        }
        result = classifier.classify(entry)
        assert result.category == EntryCategory.BOOK

    def test_classify_web_reference_with_url(self, classifier):
        from bibtex_updater.fact_checker import EntryCategory

        entry = {
            "ENTRYTYPE": "misc",
            "ID": "blog2023",
            "title": "Test Blog Post",
            "author": "Smith, John",
            "howpublished": r"\url{https://example.com/blog}",
        }
        result = classifier.classify(entry)
        assert result.category == EntryCategory.WEB_REFERENCE
        assert result.extracted_url == "https://example.com/blog"

    def test_classify_academic_url_as_academic(self, classifier):
        from bibtex_updater.fact_checker import EntryCategory

        entry = {
            "ENTRYTYPE": "misc",
            "ID": "arxiv2023",
            "title": "Test Paper",
            "author": "Smith, John",
            "howpublished": r"\url{https://arxiv.org/abs/2301.00001}",
        }
        result = classifier.classify(entry)
        assert result.category == EntryCategory.ACADEMIC

    def test_classify_article_with_doi(self, classifier):
        from bibtex_updater.fact_checker import EntryCategory

        entry = {
            "ENTRYTYPE": "article",
            "ID": "test2023",
            "title": "Test Article",
            "author": "Smith, John",
            "doi": "10.1234/test",
        }
        result = classifier.classify(entry)
        assert result.category == EntryCategory.ACADEMIC

    def test_classify_inproceedings(self, classifier):
        from bibtex_updater.fact_checker import EntryCategory

        entry = {
            "ENTRYTYPE": "inproceedings",
            "ID": "test2023",
            "title": "Test Paper",
            "author": "Smith, John",
            "booktitle": "Proceedings of NeurIPS",
        }
        result = classifier.classify(entry)
        assert result.category == EntryCategory.ACADEMIC

    def test_classify_working_paper_by_type(self, classifier):
        from bibtex_updater.fact_checker import EntryCategory

        entry = {
            "ENTRYTYPE": "techreport",
            "ID": "test2023",
            "title": "Test Report",
            "author": "Smith, John",
            "institution": "MIT",
        }
        result = classifier.classify(entry)
        assert result.category == EntryCategory.WORKING_PAPER

    def test_classify_working_paper_by_journal(self, classifier):
        from bibtex_updater.fact_checker import EntryCategory

        entry = {
            "ENTRYTYPE": "article",
            "ID": "test2023",
            "title": "Test Working Paper",
            "author": "Smith, John",
            "journal": "NBER Working Paper",
            "number": "12345",
        }
        result = classifier.classify(entry)
        assert result.category == EntryCategory.WORKING_PAPER

    def test_classify_with_eprint_as_academic(self, classifier):
        from bibtex_updater.fact_checker import EntryCategory

        entry = {
            "ENTRYTYPE": "misc",
            "ID": "arxiv2023",
            "title": "Test Preprint",
            "author": "Smith, John",
            "eprint": "2301.00001",
            "archiveprefix": "arXiv",
        }
        result = classifier.classify(entry)
        assert result.category == EntryCategory.ACADEMIC

    def test_extract_url_from_howpublished(self, classifier):
        entry = {"howpublished": r"\url{https://example.com/test}"}
        url = classifier._extract_url(entry)
        assert url == "https://example.com/test"

    def test_extract_url_from_url_field(self, classifier):
        entry = {"url": "https://example.com/test"}
        url = classifier._extract_url(entry)
        assert url == "https://example.com/test"

    def test_extract_isbn(self, classifier):
        entry = {"isbn": "978-0-123456-78-9"}
        isbn = classifier._extract_isbn(entry)
        assert isbn == "9780123456789"


# ------------- Verifier Tests -------------


class TestWebVerifier:
    """Tests for WebVerifier."""

    @pytest.fixture
    def web_verifier(self, fake_http, logger):
        from bibtex_updater.fact_checker import WebVerifier, WebVerifierConfig

        return WebVerifier(fake_http, WebVerifierConfig(), logger)

    def test_verify_returns_error_without_url(self, web_verifier):
        from bibtex_updater.fact_checker import ClassificationResult, EntryCategory, FactCheckStatus

        entry = {
            "ENTRYTYPE": "misc",
            "ID": "test",
            "title": "Test",
        }
        classification = ClassificationResult(
            category=EntryCategory.WEB_REFERENCE,
            reason="test",
            extracted_url=None,
        )
        result = web_verifier.verify(entry, classification)
        assert result.status == FactCheckStatus.API_ERROR
        assert "No URL found" in result.errors[0]


class TestBookVerifier:
    """Tests for BookVerifier."""

    @pytest.fixture
    def book_verifier(self, fake_http, logger):
        from bibtex_updater.fact_checker import BookVerifier, BookVerifierConfig

        return BookVerifier(fake_http, BookVerifierConfig(use_google_books=False), logger)

    def test_verify_returns_error_without_title(self, book_verifier):
        from bibtex_updater.fact_checker import ClassificationResult, EntryCategory, FactCheckStatus

        entry = {
            "ENTRYTYPE": "book",
            "ID": "test",
        }
        classification = ClassificationResult(
            category=EntryCategory.BOOK,
            reason="test",
        )
        result = book_verifier.verify(entry, classification)
        assert result.status == FactCheckStatus.API_ERROR
        assert "No title found" in result.errors[0]

    def test_verify_returns_book_not_found_on_no_results(self, book_verifier, fake_http):
        from bibtex_updater.fact_checker import ClassificationResult, EntryCategory, FactCheckStatus

        fake_http._request.return_value = MagicMock(status_code=200, json=lambda: {"docs": []})
        entry = {
            "ENTRYTYPE": "book",
            "ID": "test",
            "title": "Nonexistent Book",
            "author": "Nobody",
        }
        classification = ClassificationResult(
            category=EntryCategory.BOOK,
            reason="test",
        )
        result = book_verifier.verify(entry, classification)
        assert result.status == FactCheckStatus.BOOK_NOT_FOUND


class TestUnifiedFactChecker:
    """Tests for UnifiedFactChecker."""

    @pytest.fixture
    def unified_checker(self, fake_http, fake_crossref, fake_dblp, fake_s2, logger):
        from bibtex_updater.fact_checker import UnifiedFactChecker

        return UnifiedFactChecker(
            http=fake_http,
            crossref=fake_crossref,
            dblp=fake_dblp,
            s2=fake_s2,
            config=FactCheckerConfig(),
            logger=logger,
        )

    def test_classifies_and_delegates_book(self, unified_checker, fake_http):
        from bibtex_updater.fact_checker import EntryCategory

        fake_http._request.return_value = MagicMock(status_code=200, json=lambda: {"docs": []})
        entry = {
            "ENTRYTYPE": "book",
            "ID": "test2023",
            "title": "Test Book",
            "author": "Smith, John",
            "publisher": "Test Publisher",
        }
        result = unified_checker.check_entry(entry)
        assert result.category == EntryCategory.BOOK

    def test_skip_categories(self, fake_http, fake_crossref, fake_dblp, fake_s2, logger):
        from bibtex_updater.fact_checker import EntryCategory, FactCheckStatus, UnifiedFactChecker

        checker = UnifiedFactChecker(
            http=fake_http,
            crossref=fake_crossref,
            dblp=fake_dblp,
            s2=fake_s2,
            config=FactCheckerConfig(),
            logger=logger,
            skip_categories=[EntryCategory.BOOK],
        )
        entry = {
            "ENTRYTYPE": "book",
            "ID": "test2023",
            "title": "Test Book",
        }
        result = checker.check_entry(entry)
        assert result.status == FactCheckStatus.SKIPPED
        assert result.category == EntryCategory.BOOK


# ------------- Year Validation Tests -------------


class TestYearValidation:
    """Tests for year validation pre-API check."""

    def test_future_year(self, fact_checker):
        entry = {"ID": "test", "ENTRYTYPE": "article", "title": "Test Paper", "author": "Smith", "year": "2099"}
        result = fact_checker.check_entry(entry)
        assert result.status == FactCheckStatus.FUTURE_DATE

    def test_non_numeric_year(self, fact_checker):
        entry = {"ID": "test", "ENTRYTYPE": "article", "title": "Test", "author": "Smith", "year": "forthcoming"}
        result = fact_checker.check_entry(entry)
        assert result.status == FactCheckStatus.INVALID_YEAR

    def test_implausible_year(self, fact_checker):
        entry = {"ID": "test", "ENTRYTYPE": "article", "title": "Test", "author": "Smith", "year": "1700"}
        result = fact_checker.check_entry(entry)
        assert result.status == FactCheckStatus.INVALID_YEAR

    def test_valid_year_passes(self, fact_checker):
        entry = {"ID": "test", "ENTRYTYPE": "article", "title": "Test Paper", "author": "Smith", "year": "2023"}
        result = fact_checker.check_entry(entry)
        assert result.status != FactCheckStatus.FUTURE_DATE
        assert result.status != FactCheckStatus.INVALID_YEAR

    def test_missing_year_passes(self, fact_checker):
        entry = {"ID": "test", "ENTRYTYPE": "article", "title": "Test Paper", "author": "Smith"}
        result = fact_checker.check_entry(entry)
        assert result.status != FactCheckStatus.FUTURE_DATE
        assert result.status != FactCheckStatus.INVALID_YEAR

    def test_year_with_braces(self, fact_checker):
        entry = {"ID": "test", "ENTRYTYPE": "article", "title": "Test", "author": "Smith", "year": "{2023}"}
        result = fact_checker.check_entry(entry)
        assert result.status != FactCheckStatus.FUTURE_DATE
        assert result.status != FactCheckStatus.INVALID_YEAR

    def test_check_years_disabled(self, fake_crossref, fake_dblp, fake_s2, logger):
        config = FactCheckerConfig(check_years=False)
        checker = FactChecker(fake_crossref, fake_dblp, fake_s2, config, logger)
        entry = {"ID": "test", "ENTRYTYPE": "article", "title": "Test", "author": "Smith", "year": "2099"}
        result = checker.check_entry(entry)
        assert result.status != FactCheckStatus.FUTURE_DATE


# ------------- DOI Validation Tests -------------


class TestDOIValidation:
    """Tests for DOI resolution validation."""

    def test_validate_doi_no_doi(self, fact_checker):
        result = fact_checker._validate_doi({"title": "Test"})
        assert result is None

    def test_validate_doi_url_format(self, fact_checker):
        """DOI in URL format should be cleaned."""
        # We can't easily test real resolution, but verify URL cleaning
        result = fact_checker._validate_doi({"doi": "https://doi.org/10.1234/test"})
        # Should return None (network ok or caught) or DOI_NOT_FOUND
        assert result is None or result == FactCheckStatus.DOI_NOT_FOUND

    def test_check_dois_disabled(self, fake_crossref, fake_dblp, fake_s2, logger):
        config = FactCheckerConfig(check_dois=False)
        checker = FactChecker(fake_crossref, fake_dblp, fake_s2, config, logger)
        entry = {"ID": "test", "ENTRYTYPE": "article", "title": "Test", "author": "Smith", "doi": "10.9999/fake"}
        result = checker.check_entry(entry)
        assert result.status != FactCheckStatus.DOI_NOT_FOUND


# ------------- FIX D: arXiv DOI version-suffix Tests -------------


class TestArxivDoiVersionStripping:
    """FIX D: versioned arXiv DOIs must be version-stripped before resolution."""

    def test_versioned_arxiv_doi_stripped_and_resolves(self, fact_checker):
        """The versioned arXiv DOI 404s, but the version-stripped one resolves (302).

        We must NOT flag it as DOI_NOT_FOUND.
        """
        client = fact_checker.crossref.http.client

        def fake_head(url, headers=None):
            # The stripped (unversioned) DOI is what should be requested.
            assert "v1" not in url
            assert url == "https://doi.org/10.48550/arxiv.2010.11929"
            return MagicMock(status_code=302)

        client.head.side_effect = fake_head
        result = fact_checker._validate_doi({"doi": "10.48550/arXiv.2010.11929v1"})
        assert result is None

    def test_non_arxiv_doi_version_like_suffix_not_stripped(self, fact_checker):
        """A non-arXiv DOI ending in letter+digit must NOT be version-stripped."""
        client = fact_checker.crossref.http.client
        seen = {}

        def fake_head(url, headers=None):
            seen["url"] = url
            return MagicMock(status_code=200)

        client.head.side_effect = fake_head
        fact_checker._validate_doi({"doi": "10.1234/journal.v2"})
        # The trailing 'v2' must survive (non-arXiv prefix).
        assert seen["url"] == "https://doi.org/10.1234/journal.v2"

    def test_head_hostile_host_retries_with_get(self, fact_checker):
        """A 404 to HEAD that resolves on GET must NOT be DOI_NOT_FOUND."""
        client = fact_checker.crossref.http.client
        client.head.return_value = MagicMock(status_code=404)
        client.get.return_value = MagicMock(status_code=206)  # ranged GET success
        result = fact_checker._validate_doi({"doi": "10.1234/headhostile"})
        assert result is None
        client.get.assert_called_once()

    def test_genuinely_missing_doi_still_flagged(self, fact_checker):
        """404 on both HEAD and GET -> genuinely missing DOI."""
        client = fact_checker.crossref.http.client
        client.head.return_value = MagicMock(status_code=404)
        client.get.return_value = MagicMock(status_code=410)
        result = fact_checker._validate_doi({"doi": "10.1234/gone"})
        assert result == FactCheckStatus.DOI_NOT_FOUND

    def test_publisher_block_not_flagged(self, fact_checker):
        """418/403/429 are publisher blocks, not invalid DOIs."""
        client = fact_checker.crossref.http.client
        client.head.return_value = MagicMock(status_code=418)
        result = fact_checker._validate_doi({"doi": "10.1109/ieee.block"})
        assert result is None

    def test_batch_validate_strips_arxiv_version(self, processor):
        """_batch_validate_dois must also version-strip arXiv DOIs."""
        client = processor.checker.crossref.http.client
        requested = []

        def fake_head(url, headers=None):
            requested.append(url)
            return MagicMock(status_code=302)

        client.head.side_effect = fake_head
        entries = [{"ID": "vit", "doi": "10.48550/arXiv.2010.11929v1"}]
        results = processor._batch_validate_dois(entries)
        assert results["vit"] is True
        assert requested == ["https://doi.org/10.48550/arxiv.2010.11929"]


# ------------- Venue Matching Tests -------------


class TestVenueMatching:
    """Tests for venue matching with aliases (now three-valued)."""

    def test_exact_match(self):
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        result = venues_match("NeurIPS", "NeurIPS")
        assert result.outcome is MatchOutcome.MATCH
        assert result.is_confirmed is True

    def test_alias_match_neurips(self):
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        result = venues_match("NeurIPS", "Advances in Neural Information Processing Systems")
        assert result.outcome is MatchOutcome.MATCH
        assert result.score >= 0.90

    def test_alias_match_nips_neurips(self):
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        assert venues_match("NIPS", "NeurIPS").outcome is MatchOutcome.MATCH

    def test_alias_match_icml(self):
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        assert venues_match("ICML", "International Conference on Machine Learning").outcome is MatchOutcome.MATCH

    def test_different_venues(self):
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        result = venues_match("NeurIPS", "ICML")
        assert result.outcome is MatchOutcome.MISMATCH
        assert result.is_mismatch is True

    def test_empty_entry_venue(self):
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        # No claim on one side -> non-comparable (NOT a positive confirmation).
        result = venues_match("", "NeurIPS")
        assert result.outcome is MatchOutcome.NON_COMPARABLE
        assert result.is_confirmed is False

    def test_proceedings_prefix_stripped(self):
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        result = venues_match(
            "Proceedings of NeurIPS 2023",
            "Advances in Neural Information Processing Systems",
        )
        assert result.outcome is MatchOutcome.MATCH

    def test_wrong_venue_detected(self):
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        result = venues_match(
            "IEEE Conference on Computer Vision and Pattern Recognition",
            "IEEE International Conference on Computer Vision",
        )
        assert result.outcome is MatchOutcome.MISMATCH

    def test_venue_in_fact_checker(self, fact_checker):
        """Venue mismatch detected in field comparison."""
        entry = {
            "ID": "test",
            "ENTRYTYPE": "inproceedings",
            "title": "Test Paper",
            "author": "Smith, John",
            "booktitle": "NeurIPS",
            "year": "2023",
        }
        record = PublishedRecord(
            doi="10.1234/test",
            title="Test Paper",
            authors=[{"given": "John", "family": "Smith"}],
            journal="ICML",
            year=2023,
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["venue"].matches is False


class TestVenueSymmetryAndPreprint:
    """FIX E-venue: symmetric empty handling + preprint/series non-comparability.

    Non-comparable cases are now distinguished from positive confirmation: a
    blank or preprint-only record cannot CONFIRM the claimed published venue, so
    it returns NON_COMPARABLE (not MATCH and not MISMATCH).
    """

    def test_empty_api_venue_is_non_comparable_not_mismatch(self):
        """Empty API venue (arXiv-indexed record) must NOT hard-fail as mismatch."""
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        result = venues_match("NeurIPS", "")
        assert result.outcome is MatchOutcome.NON_COMPARABLE
        assert result.is_mismatch is False
        assert result.is_confirmed is False

    def test_empty_entry_venue_is_non_comparable(self):
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        assert venues_match("", "NeurIPS").outcome is MatchOutcome.NON_COMPARABLE

    def test_arxiv_api_venue_is_non_comparable(self):
        """A published-conference entry vs an arXiv API venue is non-comparable."""
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        result = venues_match("NeurIPS", "arXiv preprint arXiv:2010.11929")
        assert result.outcome is MatchOutcome.NON_COMPARABLE
        assert result.is_confirmed is False

    def test_corr_api_venue_is_non_comparable(self):
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        assert venues_match("ICML", "CoRR").outcome is MatchOutcome.NON_COMPARABLE

    def test_pmlr_series_is_non_comparable(self):
        """PMLR is an umbrella series, not a specific venue."""
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        assert venues_match("ICML", "Proceedings of Machine Learning Research").outcome is MatchOutcome.NON_COMPARABLE

    def test_genuinely_different_venues_still_mismatch(self):
        """Two populated, distinct real venues must still be detectable."""
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        assert venues_match("NeurIPS", "ICML").outcome is MatchOutcome.MISMATCH

    def test_neurips_full_name_aliases(self):
        from bibtex_updater.fact_checker import venues_match
        from bibtex_updater.matching import MatchOutcome

        for full in (
            "Conference on Neural Information Processing Systems",
            "Annual Conference on Neural Information Processing Systems",
        ):
            assert venues_match("NeurIPS", full).outcome is MatchOutcome.MATCH, full

    def test_empty_api_venue_in_field_comparison_not_mismatch(self, fact_checker):
        """An arXiv-indexed record with blank venue must NOT produce venue_mismatch.

        It is non-comparable: not a confirmed match (so ``matches`` is False), but
        also not a mismatch -- the verdict routes to UNCONFIRMED, not VENUE_MISMATCH.
        """
        from bibtex_updater.matching import MatchOutcome

        entry = {
            "ID": "test",
            "ENTRYTYPE": "inproceedings",
            "title": "Attention Is All You Need",
            "author": "Vaswani, Ashish",
            "booktitle": "NeurIPS",
            "year": "2017",
        }
        record = PublishedRecord(
            doi="10.48550/arXiv.1706.03762",
            title="Attention Is All You Need",
            authors=[{"given": "Ashish", "family": "Vaswani"}],
            journal="",  # arXiv-indexed: blank venue
            year=2017,
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        venue = comparisons["venue"]
        assert venue.outcome is MatchOutcome.NON_COMPARABLE
        assert venue.is_mismatch is False
        assert venue.matches is False  # non-comparable is NOT a positive confirmation


class TestNormalizeVenue:
    """Tests for venue normalization."""

    def test_strip_proceedings_prefix(self):
        from bibtex_updater.fact_checker import normalize_venue

        assert "neurips" in normalize_venue("Proceedings of NeurIPS 2023")

    def test_lowercase(self):
        from bibtex_updater.fact_checker import normalize_venue

        result = normalize_venue("NeurIPS")
        assert result == "neurips"

    def test_strip_year(self):
        from bibtex_updater.fact_checker import normalize_venue

        result = normalize_venue("ICML 2023")
        assert "2023" not in result


# ------------- Preprint Detection Tests -------------


class TestPreprintDetection:
    """Tests for preprint-vs-published detection."""

    def test_no_venue_claim_skips(self, fact_checker):
        entry = {"ID": "test", "ENTRYTYPE": "misc", "title": "Test", "author": "Smith", "eprint": "2301.00001"}
        result = fact_checker._check_preprint_status(entry, PublishedRecord(doi="", title="Test"))
        assert result is None

    def test_arxiv_venue_skips(self, fact_checker):
        entry = {
            "ID": "test",
            "ENTRYTYPE": "misc",
            "title": "Test",
            "author": "Smith",
            "journal": "arXiv preprint arXiv:2301.00001",
        }
        result = fact_checker._check_preprint_status(entry, PublishedRecord(doi="", title="Test"))
        assert result is None

    def test_no_identifiers_skips(self, fact_checker):
        entry = {
            "ID": "test",
            "ENTRYTYPE": "inproceedings",
            "title": "Test",
            "author": "Smith",
            "booktitle": "NeurIPS",
        }
        result = fact_checker._check_preprint_status(entry, PublishedRecord(doi="", title="Test"))
        assert result is None

    def test_preprint_only_detected(self, fake_http, fake_crossref, fake_dblp, logger):
        mock_s2_response = MagicMock(
            status_code=200,
            json=lambda: {
                "title": "Test Paper",
                "authors": [{"name": "John Smith"}],
                "venue": "",
                "year": 2023,
                "publicationTypes": [],
                "externalIds": {"ArXiv": "2301.00001"},
                "publicationVenue": None,
                "url": "https://arxiv.org/abs/2301.00001",
            },
        )
        fake_http._request.return_value = mock_s2_response
        s2 = SemanticScholarClient(fake_http)
        checker = FactChecker(fake_crossref, fake_dblp, s2, FactCheckerConfig(), logger)
        entry = {
            "ID": "test",
            "ENTRYTYPE": "inproceedings",
            "title": "Test Paper",
            "author": "Smith, John",
            "booktitle": "NeurIPS",
            "eprint": "2301.00001",
        }
        result = checker._check_preprint_status(entry, PublishedRecord(doi="", title="Test Paper"))
        assert result == FactCheckStatus.PREPRINT_ONLY

    def test_published_paper_passes(self, fake_http, fake_crossref, fake_dblp, logger):
        mock_s2_response = MagicMock(
            status_code=200,
            json=lambda: {
                "title": "Test Paper",
                "authors": [{"name": "John Smith"}],
                "venue": "NeurIPS",
                "year": 2023,
                "publicationTypes": ["Conference"],
                "externalIds": {"ArXiv": "2301.00001", "DOI": "10.1234/test"},
                "publicationVenue": {"name": "NeurIPS"},
                "url": "https://doi.org/10.1234/test",
            },
        )
        fake_http._request.return_value = mock_s2_response
        s2 = SemanticScholarClient(fake_http)
        checker = FactChecker(fake_crossref, fake_dblp, s2, FactCheckerConfig(), logger)
        entry = {
            "ID": "test",
            "ENTRYTYPE": "inproceedings",
            "title": "Test Paper",
            "author": "Smith, John",
            "booktitle": "NeurIPS",
            "doi": "10.1234/test",
        }
        result = checker._check_preprint_status(entry, PublishedRecord(doi="10.1234/test", title="Test Paper"))
        assert result is None


# ------------- Streaming JSONL Tests -------------


class TestStreamingJSONL:
    """Tests for streaming JSONL output."""

    def test_streaming_writes_incrementally(self, processor, tmp_path):
        """When jsonl_path is provided, results are written incrementally."""
        jsonl_file = tmp_path / "output.jsonl"
        entries = [
            {"ID": "a", "ENTRYTYPE": "article", "title": "Test A", "author": "Smith"},
            {"ID": "b", "ENTRYTYPE": "article", "title": "Test B", "author": "Doe"},
        ]
        results = processor.process_entries(entries, jsonl_path=str(jsonl_file))
        assert len(results) == 2
        lines = jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 2
        import json

        for line in lines:
            data = json.loads(line)
            assert "key" in data
            assert "status" in data

    def test_no_jsonl_path_works(self, processor):
        """Without jsonl_path, behavior is unchanged."""
        entries = [{"ID": "a", "ENTRYTYPE": "article", "title": "Test A", "author": "Smith"}]
        results = processor.process_entries(entries)
        assert len(results) == 1

    def test_partial_results_survive(self, tmp_path, fake_crossref, fake_dblp, fake_s2, logger):
        """Partial results survive if processing raises mid-way."""

        class FailingChecker:
            def check_entry(self, entry):
                if entry.get("ID") == "fail":
                    raise RuntimeError("Simulated failure")
                return FactCheckResult(
                    entry.get("ID", "?"),
                    "article",
                    FactCheckStatus.NOT_FOUND,
                    0.0,
                    {},
                    None,
                    [],
                    [],
                    [],
                )

        proc = FactCheckProcessor(FailingChecker(), logger)
        jsonl_file = tmp_path / "partial.jsonl"
        entries = [
            {"ID": "ok1", "ENTRYTYPE": "article", "title": "Good", "author": "A"},
            {"ID": "ok2", "ENTRYTYPE": "article", "title": "Good2", "author": "B"},
            {"ID": "fail", "ENTRYTYPE": "article", "title": "Bad", "author": "C"},
        ]
        # Error recovery: failed entries get API_ERROR status instead of being dropped
        results = proc.process_entries(entries, jsonl_path=str(jsonl_file))
        # All 3 entries should have results (failed entry gets API_ERROR)
        assert len(results) == 3
        error_results = [r for r in results if r.status == FactCheckStatus.API_ERROR]
        assert len(error_results) == 1
        assert error_results[0].entry_key == "fail"
        # All entries should have been flushed to JSONL (including the error one)
        lines = jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 3


# ------------- New Status Tests -------------


class TestNewStatuses:
    """Verify new FactCheckStatus values exist."""

    def test_future_date_status(self):
        assert FactCheckStatus.FUTURE_DATE.value == "future_date"

    def test_invalid_year_status(self):
        assert FactCheckStatus.INVALID_YEAR.value == "invalid_year"

    def test_doi_not_found_status(self):
        assert FactCheckStatus.DOI_NOT_FOUND.value == "doi_not_found"

    def test_preprint_only_status(self):
        assert FactCheckStatus.PREPRINT_ONLY.value == "preprint_only"

    def test_published_version_exists_status(self):
        assert FactCheckStatus.PUBLISHED_VERSION_EXISTS.value == "published_version_exists"


# ------------- S2 API Key Tests -------------


class TestSemanticScholarGetPaper:
    """Tests for S2 get_paper method."""

    def test_get_paper_returns_none_on_error(self, fake_http):
        fake_http._request.side_effect = Exception("Network error")
        client = SemanticScholarClient(fake_http)
        result = client.get_paper("DOI:10.1234/test")
        assert result is None

    def test_get_paper_returns_none_on_404(self, fake_http):
        fake_http._request.return_value = MagicMock(status_code=404)
        client = SemanticScholarClient(fake_http)
        result = client.get_paper("DOI:10.1234/test")
        assert result is None

    def test_get_paper_returns_data(self, fake_http):
        fake_http._request.return_value = MagicMock(
            status_code=200,
            json=lambda: {"title": "Test", "venue": "NeurIPS"},
        )
        client = SemanticScholarClient(fake_http)
        result = client.get_paper("DOI:10.1234/test")
        assert result is not None
        assert result["title"] == "Test"
