"""Tests for reference_fact_checker module."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from bib_utils import PublishedRecord
from reference_fact_checker import (
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

    def test_compare_missing_venue_allowed(self, fact_checker):
        entry = {"title": "Test", "author": "Author"}  # No journal
        record = PublishedRecord(doi="", title="Test", journal="Some Journal")
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["venue"].matches is True


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

    def test_hallucinated_low_score(self, fact_checker):
        comparisons = {"title": FieldComparison("title", "A", "B", 0.3, False)}
        status = fact_checker._determine_status(0.35, comparisons, [])
        assert status == FactCheckStatus.HALLUCINATED

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


class TestFactCheckerCheckEntry:
    """Tests for FactChecker.check_entry method."""

    def test_entry_without_title(self, fact_checker):
        entry = {"ID": "notitle", "ENTRYTYPE": "article", "author": "Smith"}
        result = fact_checker.check_entry(entry)
        assert result.status == FactCheckStatus.API_ERROR
        assert "No title" in result.errors[0]

    def test_entry_not_found(self, fact_checker, sample_entry):
        # All API clients return empty results by default
        result = fact_checker.check_entry(sample_entry)
        assert result.status == FactCheckStatus.NOT_FOUND
        assert len(result.api_sources_queried) == 3


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
        assert summary["problematic_count"] == 2

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
