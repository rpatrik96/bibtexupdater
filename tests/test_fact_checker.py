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
    build_verification_result,
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

    def test_compare_year_preprint_record_does_not_mismatch(self, fact_checker):
        """Regression (ea48d1a613c3): a correct citation of a PUBLISHED version
        whose matched record is the arXiv PREPRINT twin (posted years earlier)
        must not read as a year_mismatch -- a preprint cannot refute a published
        year -> NON_COMPARABLE."""
        from bibtex_updater.matching import MatchOutcome

        entry = {"title": "Test", "year": "2022", "author": "Author"}
        record = PublishedRecord(doi="10.48550/arXiv.1910.03834", title="Test", year=2019)
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["year"].outcome is MatchOutcome.NON_COMPARABLE
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

    def test_gross_author_mismatch_with_incidental_substitution_stays_author_mismatch(self, fact_checker):
        """Regression: refchecker2024 incident (hallmark-paper, 2026-06).

        A fabricated author list (1 of 10 real) collides with the canonical
        roster on two frequent surnames, so the given-name audit records
        SUBSTITUTION findings at those positions -- but the author note is NOT
        the audit's matching-surnames escalation. The verdict must stay
        AUTHOR_MISMATCH (problematic), not the benign-sounding
        GIVEN_NAME_SUBSTITUTION that buried the fabrication.
        """
        author = FieldComparison(
            "author",
            "Hu, Xiangkun and Gao, Dongyu and Liu, Zhengying and Zhang, Michael R.",
            "Xiangkun Hu and Dongyu Ru and Pengfei Liu and Yue Zhang",
            0.16,
            False,
            given_name_findings=[
                {"position": 0, "variety": "given_exact", "entry_given": "Xiangkun", "record_given": "Xiangkun"},
                {
                    "position": 2,
                    "variety": "given_name_substitution",
                    "entry_given": "Zhengying",
                    "record_given": "Pengfei",
                },
                {
                    "position": 3,
                    "variety": "given_name_substitution",
                    "entry_given": "Michael R.",
                    "record_given": "Yue",
                },
            ],
        )
        comparisons = {
            "title": FieldComparison("title", "RefChecker", "RefChecker", 1.0, True),
            "author": author,
            "year": FieldComparison("year", "2024", "2024", 1.0, True),
            "venue": FieldComparison("venue", "NAACL", "NAACL", 1.0, True),
        }
        status = fact_checker._determine_status(0.80, comparisons, ["crossref"])
        assert status == FactCheckStatus.AUTHOR_MISMATCH

    def test_escalated_given_name_substitution_still_routes(self, fact_checker):
        """The audit's matching-surnames escalation (its note marks the path)
        keeps its dedicated root-cause status."""
        author = FieldComparison(
            "author",
            "Zhao, Yue and Smith, John",
            "Yujing Zhao and John Smith",
            0.9,
            False,
            note="Given-name substitution on matching surnames (likely a wrong author)",
            given_name_findings=[
                {"position": 0, "variety": "given_name_substitution", "entry_given": "Yue", "record_given": "Yujing"},
                {"position": 1, "variety": "given_exact", "entry_given": "John", "record_given": "John"},
            ],
        )
        comparisons = {
            "title": FieldComparison("title", "A", "A", 1.0, True),
            "author": author,
            "year": FieldComparison("year", "2021", "2021", 1.0, True),
            "venue": FieldComparison("venue", "J", "J", 1.0, True),
        }
        status = fact_checker._determine_status(0.85, comparisons, ["crossref"])
        assert status == FactCheckStatus.GIVEN_NAME_SUBSTITUTION

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
        # queries five sources: crossref -> openalex -> dblp -> openreview ->
        # semanticscholar (OpenAlex and OpenReview are lazily built from the
        # shared crossref.http mock).
        # FIX X4: when every primary source returns nothing usable, the
        # relaxed-author retrieval fallback runs (title-only retry on
        # Crossref + OpenAlex) and appends two fallback source names.
        result = fact_checker.check_entry(sample_entry)
        assert result.status == FactCheckStatus.NOT_FOUND
        assert result.api_sources_queried == [
            "crossref",
            "openalex",
            "dblp",
            "openreview",
            "semanticscholar",
            "crossref-fallback",
            "openalex-fallback",
        ]

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


# ------------- Thread-safety of per-entry state -------------


class TestCheckEntryThreadSafety:
    """check_entry runs concurrently across a ThreadPoolExecutor (see
    FactCheckProcessor.process_entries). Per-entry verification state (the
    cross-source author intersection and the per-source records) must therefore
    live on each FactCheckResult, NOT on the shared FactChecker instance, or
    concurrent entries clobber each other and produce nondeterministic verdicts.
    """

    @staticmethod
    def _matching_record(entry: dict) -> PublishedRecord:
        """Build a record that matches an entry exactly (so it VERIFIES)."""
        # surname -> {"given","family"} so the author intersection is non-trivial
        # and entry-specific.
        family = entry["author"].split(",")[0].strip()
        return PublishedRecord(
            doi=f"10.9999/{entry['ID']}",
            title=entry["title"],
            authors=[{"given": "A", "family": family}],
            journal=entry["journal"],
            year=int(entry["year"]),
        )

    def _install_per_entry_cascade(self, checker, monkeypatch, barrier=None):
        """Stub the cascade so each entry resolves to its OWN matching record.

        If ``barrier`` is given, every worker blocks on it inside the cascade so
        all entries are guaranteed to be *interleaved* inside check_entry at the
        same time -- the exact window where shared-instance state would race.
        """

        def fake_cascade(entry, query, sq, sh, errors):
            sq.append("crossref")
            sh.append("crossref")
            rec = self._matching_record(entry)
            if barrier is not None:
                # Force all workers to sit inside check_entry simultaneously.
                barrier.wait(timeout=10)
            return [(0.99, rec, "crossref")]

        monkeypatch.setattr(checker, "_query_cascade", fake_cascade)
        monkeypatch.setattr(checker, "_query_arxiv_by_id", lambda *a, **k: [])

    @staticmethod
    def _entry(i: int) -> dict:
        return {
            "ID": f"entry{i}",
            "ENTRYTYPE": "article",
            "title": f"Unique Paper Title Number {i}",
            "author": f"Surname{i}, Given",
            "journal": f"Journal {i}",
            "year": str(2000 + (i % 20)),
        }

    def test_no_per_entry_state_left_on_instance(self, fact_checker, monkeypatch):
        """After check_entry, nothing entry-specific lives on the shared self."""
        self._install_per_entry_cascade(fact_checker, monkeypatch)
        result = fact_checker.check_entry(self._entry(1))
        assert result.status == FactCheckStatus.VERIFIED
        # The old racy attributes must be gone entirely.
        assert not hasattr(fact_checker, "last_author_intersection")
        assert not hasattr(fact_checker, "last_source_records")
        # Per-entry state rides on the result instead.
        assert result.source_records  # populated
        assert "crossref" in result.source_records
        assert result.author_intersection is not None

    def test_result_carries_its_own_records(self, fact_checker, monkeypatch):
        """Each result's source_records/intersection describe ITS entry."""
        self._install_per_entry_cascade(fact_checker, monkeypatch)
        r3 = fact_checker.check_entry(self._entry(3))
        r7 = fact_checker.check_entry(self._entry(7))
        # The per-source record for each result is the one matching its entry.
        assert r3.source_records["crossref"].doi == "10.9999/entry3"
        assert r7.source_records["crossref"].doi == "10.9999/entry7"
        # A sequential second call must not have mutated the first result.
        assert r3.source_records["crossref"].doi == "10.9999/entry3"

    def test_concurrent_check_entry_no_cross_contamination(self, fact_checker, monkeypatch):
        """Interleaved concurrent check_entry calls must not cross-contaminate
        the per-entry records, the intersection, or the rich VerificationResult.
        """
        import concurrent.futures
        import threading

        n = 16
        entries = [self._entry(i) for i in range(n)]
        # Barrier forces all n workers to be *inside* check_entry simultaneously,
        # maximizing the chance a shared-state bug would surface.
        barrier = threading.Barrier(n)
        self._install_per_entry_cascade(fact_checker, monkeypatch, barrier=barrier)

        def run(entry):
            res = fact_checker.check_entry(entry)
            rich = build_verification_result(res)
            return entry["ID"], res, rich

        out: dict[str, tuple] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            for key, res, rich in ex.map(run, entries):
                out[key] = (res, rich)

        assert len(out) == n
        for i in range(n):
            key = f"entry{i}"
            res, rich = out[key]
            assert res.status == FactCheckStatus.VERIFIED, key
            # Each result's source record matches its OWN entry (the bug would
            # leave a sibling entry's record here).
            assert res.source_records["crossref"].doi == f"10.9999/{key}", key
            # The rich result, built from res alone (no self.last_*), is entry-
            # specific: it reflects this entry's own matched metadata.
            assert rich.bibtex_key == key
            assert rich.matched_metadata is not None
            assert rich.matched_metadata["doi"] == f"10.9999/{key}", key

        # And the shared instance still holds no per-entry leftovers.
        assert not hasattr(fact_checker, "last_author_intersection")
        assert not hasattr(fact_checker, "last_source_records")

    def test_arxiv_record_cache_is_thread_safe(self, fact_checker, monkeypatch):
        """The cross-entry arXiv memo is guarded: concurrent lookups of distinct
        IDs all populate the cache without losing entries, the fetch runs outside
        the lock, and a given ID is fetched at most once.
        """
        import concurrent.futures
        import threading

        # Real ArxivClient is None on the fixture; install a fake that records
        # how many times each ID is fetched (a cache miss == one fetch).
        fetch_counts: dict[str, int] = {}
        counts_lock = threading.Lock()

        class _FakeArxiv:
            def fetch_atom(self, arxiv_id: str) -> str:
                with counts_lock:
                    fetch_counts[arxiv_id] = fetch_counts.get(arxiv_id, 0) + 1
                # Return a minimal value; parsing is monkeypatched below.
                return f"<atom>{arxiv_id}</atom>"

        fact_checker.arxiv = _FakeArxiv()
        # Parse step: map the xml back to a per-id record (no real XML needed).
        monkeypatch.setattr(
            "bibtex_updater.fact_checker.arxiv_atom_to_record",
            lambda xml: PublishedRecord(doi=None, title=xml, authors=[]),
        )

        ids = [f"2401.{i:05d}" for i in range(8)]
        # Hammer each ID from several threads at once -> exercises the double-
        # checked insert: at most one fetch per ID despite the concurrency.
        tasks = ids * 6

        def run(arxiv_id):
            return arxiv_id, fact_checker._arxiv_record(arxiv_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
            results = list(ex.map(run, tasks))

        # Every lookup returned the correct per-id record.
        for arxiv_id, rec in results:
            assert rec is not None
            assert rec.title == f"<atom>{arxiv_id}</atom>"
        # The cache holds exactly the distinct IDs, none lost to races.
        assert set(fact_checker._arxiv_record_cache.keys()) == set(ids)
        # Each distinct ID was fetched at most once despite 6x concurrent hits.
        for arxiv_id in ids:
            assert fetch_counts[arxiv_id] == 1, (arxiv_id, fetch_counts[arxiv_id])


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
        assert summary["verified_count"] + summary["abstained_count"] + summary["problematic_count"] == summary["total"]

    def test_given_name_substitution_counts_as_problematic(self, processor):
        """GIVEN_NAME_SUBSTITUTION is positive wrong-author evidence: it must
        land in the PROBLEMATIC bucket (and so trip the strict CI gate), in
        line with calibration's CLEARLY-PROBLEM class for the status."""

        def _r(key, status):
            return FactCheckResult(key, "article", status, 0.7, {}, None, ["crossref"], ["crossref"], [])

        results = [
            _r("v", FactCheckStatus.VERIFIED),
            _r("g", FactCheckStatus.GIVEN_NAME_SUBSTITUTION),
        ]
        summary = processor.generate_summary(results)
        assert summary["problematic_count"] == 1
        assert summary["verified_count"] + summary["abstained_count"] + summary["problematic_count"] == summary["total"]

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


class TestPreprintDoiVenue:
    """A matched record that IS a preprint (arXiv/bioRxiv DOI) cannot confirm the
    claimed *published* venue, even when its venue STRING is an unrecognized
    repository name (e.g. 'UvA-DARE') that the string-based preprint heuristic
    misses. Such a record routes venue to NON_COMPARABLE, never a MISMATCH.
    """

    def test_arxiv_doi_record_with_repository_venue_string_is_non_comparable(self, fact_checker):
        # Regression: 'Adam' (entry: ICLR 2015) matched its arXiv-DOI preprint
        # whose Crossref/OpenAlex 'journal' came back as the institutional repo
        # 'UvA-DARE (University of Amsterdam)'. is_preprint_or_series_venue does
        # NOT recognize that string, so the old code fell through to a venue
        # MISMATCH against 'ICLR'. The arXiv DOI is the authoritative preprint
        # signal -> NON_COMPARABLE.
        entry = {
            "title": "Adam: A Method for Stochastic Optimization",
            "author": "Kingma, Diederik P. and Ba, Jimmy",
            "booktitle": "International Conference on Learning Representations (ICLR)",
            "year": "2015",
        }
        record = PublishedRecord(
            doi="10.48550/arxiv.1412.6980",
            title="Adam: A Method for Stochastic Optimization",
            authors=[
                {"given": "Diederik P.", "family": "Kingma"},
                {"given": "Jimmy", "family": "Ba"},
            ],
            journal="UvA-DARE (University of Amsterdam)",
            year=2014,
            structured_names=True,
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        venue = comparisons["venue"]
        assert venue.outcome is MatchOutcome.NON_COMPARABLE
        assert venue.is_mismatch is False
        assert venue.matches is False

    def test_biorxiv_doi_record_is_non_comparable(self, fact_checker):
        entry = {"title": "T", "author": "Smith, John", "journal": "Nature", "year": "2021"}
        record = PublishedRecord(
            doi="10.1101/2021.01.01.123456",
            title="T",
            authors=[{"given": "John", "family": "Smith"}],
            journal="Some Repository",
            year=2021,
            structured_names=True,
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["venue"].outcome is MatchOutcome.NON_COMPARABLE

    def test_published_doi_record_with_different_venue_still_mismatches(self, fact_checker):
        # Guard: a NON-preprint (real publisher) DOI with a genuinely different,
        # populated venue must STILL be a venue MISMATCH. Year matches here so the
        # different-edition rule does not apply.
        entry = {"title": "T", "author": "Smith, John", "booktitle": "NeurIPS", "year": "2020"}
        record = PublishedRecord(
            doi="10.1145/3422622",
            title="T",
            authors=[{"given": "John", "family": "Smith"}],
            journal="Communications of the ACM",
            year=2020,
            structured_names=True,
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["venue"].is_mismatch is True


class TestYearNonComparable:
    """An empty or unparseable year on either side cannot confirm OR refute the
    claimed year -> NON_COMPARABLE (abstention), not a YEAR_MISMATCH.
    """

    def test_empty_record_year_is_non_comparable_not_mismatch(self, fact_checker):
        # Regression: 'DDPM' matched a DBLP 'CoRR 2020' record carrying no year.
        # The old code set year_matches=False -> read as a YEAR_MISMATCH.
        entry = {
            "title": "Denoising Diffusion Probabilistic Models",
            "author": "Ho, Jonathan",
            "booktitle": "Advances in Neural Information Processing Systems (NeurIPS)",
            "year": "2020",
        }
        record = PublishedRecord(
            doi="",
            title="Denoising Diffusion Probabilistic Models",
            authors=[{"given": "Jonathan", "family": "Ho"}],
            journal="CoRR 2020",
            year=None,
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        year = comparisons["year"]
        assert year.outcome is MatchOutcome.NON_COMPARABLE
        assert year.is_mismatch is False

    def test_no_entry_year_claim_is_vacuous_match(self, fact_checker):
        entry = {"title": "T", "author": "A"}  # no year claimed
        record = PublishedRecord(doi="", title="T", year=2020)
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["year"].outcome is MatchOutcome.MATCH
        assert comparisons["year"].matches is True

    def test_small_year_error_same_venue_still_mismatches(self, fact_checker):
        # Guard: a genuine small year error (both years present, beyond tolerance,
        # gap below the different-edition threshold) on an otherwise-matching
        # record must STILL flag YEAR_MISMATCH.
        entry = {"title": "T", "author": "A", "year": "2010", "journal": "Nature"}
        record = PublishedRecord(doi="10.1/x", title="T", journal="Nature", year=2013)
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["year"].is_mismatch is True


class TestDifferentEditionAbstains:
    """A matched record with the SAME title but published in a substantially
    different year, whose claimed venue is NOT positively confirmed, is almost
    certainly a different edition/reprint of the same work (or a same-title
    decoy) -- it can neither confirm nor refute the entry. Abstain (UNCONFIRMED),
    do not flag a *_MISMATCH.
    """

    def test_identical_title_large_year_gap_blank_venue_abstains(self, fact_checker):
        # Regression: 'Attention Is All You Need' (entry: NeurIPS 2017) matched a
        # 2025 same-title item with a blank venue -> old code: YEAR_MISMATCH.
        entry = {
            "title": "Attention Is All You Need",
            "author": "Vaswani, Ashish",
            "booktitle": "Advances in Neural Information Processing Systems (NeurIPS)",
            "year": "2017",
        }
        record = PublishedRecord(
            doi="10.65215/2q58a426",
            title="Attention Is All You Need",
            authors=[{"given": "Ashish", "family": "Vaswani"}],
            journal="",
            year=2025,
            structured_names=True,
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        assert comparisons["year"].is_mismatch is False
        status = fact_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status == FactCheckStatus.UNCONFIRMED

    def test_reprint_in_different_venue_abstains_not_partial_match(self, fact_checker):
        # Regression: 'GAN' (entry: NeurIPS 2014) matched the CACM 2020 reprint of
        # the same work (title 'Nets'->'networks' near-miss, venue CACM, year 2020)
        # -> old code: PARTIAL_MATCH. Same work, different edition -> abstain.
        entry = {
            "title": "Generative Adversarial Nets",
            "author": "Goodfellow, Ian",
            "booktitle": "Advances in Neural Information Processing Systems (NeurIPS)",
            "year": "2014",
        }
        record = PublishedRecord(
            doi="10.1145/3422622",
            title="Generative adversarial networks",
            authors=[{"given": "Ian", "family": "Goodfellow"}],
            journal="Communications of the ACM",
            year=2020,
            structured_names=True,
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        status = fact_checker._determine_status(0.90, comparisons, ["crossref"])
        assert status == FactCheckStatus.UNCONFIRMED

    def test_same_venue_large_year_gap_still_flags(self, fact_checker):
        # Guard: when the venue DOES positively match, a large year gap is NOT a
        # different-edition signature -> it remains a YEAR_MISMATCH (the entry's
        # year claim is genuinely contradicted in the same venue).
        entry = {"title": "T", "author": "Smith, John", "journal": "Nature", "year": "2008"}
        record = PublishedRecord(
            doi="10.1/x",
            title="T",
            authors=[{"given": "John", "family": "Smith"}],
            journal="Nature",
            year=2020,
            structured_names=True,
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        status = fact_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status == FactCheckStatus.YEAR_MISMATCH

    def test_wrong_author_still_flags_despite_year_gap(self, fact_checker):
        # Guard: the different-edition rule only neutralizes year/venue/near-miss
        # title -- a genuinely wrong lead author is still positive evidence.
        entry = {
            "title": "Attention Is All You Need",
            "author": "Imposter, Alice",
            "booktitle": "NeurIPS",
            "year": "2017",
        }
        record = PublishedRecord(
            doi="10.65215/2q58a426",
            title="Attention Is All You Need",
            authors=[{"given": "Ashish", "family": "Vaswani"}],
            journal="",
            year=2025,
            structured_names=True,
        )
        comparisons = fact_checker._compare_all_fields(entry, record)
        status = fact_checker._determine_status(0.95, comparisons, ["crossref"])
        assert status == FactCheckStatus.AUTHOR_MISMATCH


class TestGivenNameSubstitutionRouting:
    """End-to-end (via _compare_all_fields + _determine_status): a given-name
    substitution on matching surnames, against an order-reliable structured
    record, escalates to the distinct GIVEN_NAME_SUBSTITUTION status; an
    initials-style correct citation is NOT flagged; an unstructured record never
    escalates.
    """

    REAL_AUTHORS = [
        {"given": "Durmus", "family": "Acar"},
        {"given": "Yue", "family": "Zhao"},
        {"given": "Ramon", "family": "Navarro"},
        {"given": "Matthew", "family": "Mattina"},
    ]
    TITLE = "Federated Learning Based on Dynamic Regularization"

    def _record(self, order_reliable=True, structured=True):
        return PublishedRecord(
            doi="10.1/x",
            title=self.TITLE,
            authors=self.REAL_AUTHORS,
            year=2021,
            order_reliable=order_reliable,
            structured_names=structured,
        )

    def test_substitution_routes_to_given_name_substitution(self, fact_checker):
        entry = {
            "title": self.TITLE,
            "author": "Durmus Acar and Yujing Zhao and Rafael Navarro and Matthew Mattina",
            "year": "2021",  # no venue claim -> venue vacuously confirmed
        }
        comps = fact_checker._compare_all_fields(entry, self._record())
        assert comps["author"].is_mismatch is True
        findings = comps["author"].given_name_findings or []
        assert any(f["variety"] == "given_name_substitution" for f in findings)
        status = fact_checker._determine_status(0.95, comps, ["crossref"])
        assert status == FactCheckStatus.GIVEN_NAME_SUBSTITUTION

    def test_initials_style_correct_citation_verifies(self, fact_checker):
        entry = {
            "title": self.TITLE,
            "author": "D. Acar and Y. Zhao and R. Navarro and M. Mattina",
            "year": "2021",
        }
        comps = fact_checker._compare_all_fields(entry, self._record())
        assert comps["author"].is_mismatch is False
        assert fact_checker._determine_status(0.95, comps, ["crossref"]) == FactCheckStatus.VERIFIED

    def test_unstructured_but_order_reliable_record_escalates(self, fact_checker):
        # Loosened gate (d67418): a DBLP-style record (order-reliable, synthesized
        # names) now drives escalation -- the substitution surfaces via this source.
        entry = {
            "title": self.TITLE,
            "author": "Durmus Acar and Yujing Zhao and Rafael Navarro and Matthew Mattina",
            "year": "2021",
        }
        comps = fact_checker._compare_all_fields(entry, self._record(structured=False))
        assert comps["author"].is_mismatch is True
        assert fact_checker._determine_status(0.95, comps, ["dblp"]) == FactCheckStatus.GIVEN_NAME_SUBSTITUTION

    def test_not_order_reliable_record_does_not_escalate(self, fact_checker):
        # Semantic Scholar (not order_reliable) stays excluded -> surname-level MATCH.
        entry = {
            "title": self.TITLE,
            "author": "Durmus Acar and Yujing Zhao and Rafael Navarro and Matthew Mattina",
            "year": "2021",
        }
        comps = fact_checker._compare_all_fields(entry, self._record(structured=False, order_reliable=False))
        assert comps["author"].is_mismatch is False


class TestDoiOrgRejection:
    """doi.org's own rejection of a DOI: 404/410, or a 400 returned with no
    redirect (a malformed / unregistered DOI like a fabricated '10.77771/...').
    A 400 AFTER a redirect is a downstream publisher quirk, not doi.org's verdict.
    """

    def _resp(self, status, redirected=False):
        from types import SimpleNamespace

        return SimpleNamespace(status_code=status, history=([object()] if redirected else []))

    def test_404_and_410_are_rejected(self):
        from bibtex_updater.fact_checker import _doiorg_rejects_doi

        assert _doiorg_rejects_doi(self._resp(404)) is True
        assert _doiorg_rejects_doi(self._resp(410)) is True

    def test_doiorg_direct_400_is_rejected(self):
        # Fabricated/malformed DOI: doi.org returns 400 directly (no redirect).
        from bibtex_updater.fact_checker import _doiorg_rejects_doi

        assert _doiorg_rejects_doi(self._resp(400, redirected=False)) is True

    def test_post_redirect_400_is_not_rejected(self):
        # A 400 after doi.org's 302 is a publisher quirk, not an invalid DOI.
        from bibtex_updater.fact_checker import _doiorg_rejects_doi

        assert _doiorg_rejects_doi(self._resp(400, redirected=True)) is False

    def test_redirect_and_blocks_resolve(self):
        from bibtex_updater.fact_checker import _doiorg_rejects_doi

        for code in (200, 302, 403, 418, 429):
            assert _doiorg_rejects_doi(self._resp(code, redirected=True)) is False, code


class TestResolveWhatCanBeResolved:
    """General principle: do not stop (short-circuit the cascade) while a claimed
    field is still unconfirmed and a remaining source could confirm it; and among
    equally-good title/author matches, prefer the candidate that confirms the
    MOST claimed fields. A DOI-less conference paper (Attention/NeurIPS) matches a
    preprint perfectly on title+author but the preprint cannot confirm the venue;
    the proceedings record from DBLP/OpenReview can, so the cascade must reach it
    and selection must prefer it -> VERIFIED, not could-not-verify.
    """

    def _entry(self):
        return {
            "title": "Attention Is All You Need",
            "author": "Vaswani, Ashish",
            "booktitle": "Advances in Neural Information Processing Systems (NeurIPS)",
            "year": "2017",
        }

    def _preprint(self):
        return PublishedRecord(
            doi="10.48550/arxiv.1706.03762",
            title="Attention Is All You Need",
            authors=[{"given": "Ashish", "family": "Vaswani"}],
            journal="arXiv (Cornell University)",
            year=2017,
            structured_names=True,
        )

    def _proceedings(self):
        return PublishedRecord(
            doi="",
            title="Attention Is All You Need",
            authors=[{"given": "Ashish", "family": "Vaswani"}],
            journal="NeurIPS",
            year=2017,
            structured_names=True,
        )

    def test_preprint_alone_is_not_full_confirmation(self, fact_checker):
        cands = [(1.0, self._preprint(), "crossref")]
        assert fact_checker._has_full_confirmation(self._entry(), cands) is False

    def test_proceedings_record_yields_full_confirmation(self, fact_checker):
        cands = [(1.0, self._preprint(), "crossref"), (0.99, self._proceedings(), "dblp")]
        assert fact_checker._has_full_confirmation(self._entry(), cands) is True

    def test_selection_prefers_venue_confirming_record_over_preprint(self, fact_checker):
        cands = [(1.0, self._preprint(), "crossref"), (0.99, self._proceedings(), "dblp")]
        score, rec, src = fact_checker._select_best_candidate(self._entry(), cands)
        assert src == "dblp"
        comps = fact_checker._compare_all_fields(self._entry(), rec)
        status = fact_checker._determine_status(score, comps, ["crossref", "dblp"])
        assert status == FactCheckStatus.VERIFIED

    def test_fake_venue_never_full_confirmation(self, fact_checker):
        # Leak guard: a hallucinated published-venue claim whose only real record
        # is a preprint never reaches full confirmation -> the cascade's early
        # stop can never mint a VERIFIED for it.
        entry = {"title": "Real Title", "author": "Real, Author", "booktitle": "NeurIPS", "year": "2020"}
        preprint = PublishedRecord(
            doi="10.48550/arxiv.2003.00001",
            title="Real Title",
            authors=[{"given": "Author", "family": "Real"}],
            journal="arXiv (Cornell University)",
            year=2020,
            structured_names=True,
        )
        assert fact_checker._has_full_confirmation(entry, [(1.0, preprint, "crossref")]) is False

    def test_selection_does_not_promote_low_score_paper(self, fact_checker):
        # A clearly different (low title+author) record must NOT be promoted over
        # the genuine top match just because it shares a field.
        entry = self._entry()
        good = self._proceedings()
        unrelated = PublishedRecord(
            doi="10.1/x",
            title="A Completely Different Paper About Something Else",
            authors=[{"given": "Ashish", "family": "Vaswani"}],
            journal="NeurIPS",
            year=2017,
            structured_names=True,
        )
        cands = [(0.99, good, "dblp"), (0.40, unrelated, "s2")]
        _score, rec, _src = fact_checker._select_best_candidate(entry, cands)
        assert rec is good


class TestOrderReliablePreferenceOverArxivOnly:
    """Regression for the Least-to-Most (db9a596a4d3f) shape introduced by X2
    (commit a7c95e9). ``_arxiv_id_from_entry`` now feeds the arXiv API record
    into the candidate pool for arXiv-DataCite-DOI entries. That arXiv record
    carries ``order_reliable=False`` (the arXiv public listing doesn't expose a
    canonical author ordering for downstream comparison) and its preprint venue
    routes to ``NON_COMPARABLE`` -- which means the confirmation-key tiebreak
    awards it FEWER hard mismatches than a tied DBLP/OpenReview record whose
    published venue contradicts an entry venue claim. The arXiv-only candidate
    wins selection, the order-gated ``given_name_position_audit`` silently
    abstains (it requires ``order_reliable=True``), and the entry's lead-author
    substitution ('Shunyu Zhou' vs canonical 'Denny Zhou') disappears -- the
    entry routes to UNCONFIRMED instead of GIVEN_NAME_SUBSTITUTION.

    Fix: among candidates tied at the top of the title+author score (within the
    narrow ``_ORDER_RELIABLE_PREFERENCE_BAND``), an ``order_reliable`` candidate
    is preferred over an order-unreliable one regardless of the
    confirmation-key tiebreak. The audit then runs against the right record.
    """

    def _entry(self) -> dict[str, str]:
        # Mirrors the Least-to-Most defect shape: entry venue = ICLR, an arXiv
        # API hit will route its own venue to NON_COMPARABLE so the
        # confirmation tiebreak prefers it over a structured competitor.
        return {
            "title": "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models",
            "author": "Shunyu Zhou and Denny Zhou",
            "booktitle": "ICLR",
            "year": "2023",
        }

    def _structured_dblp(self) -> PublishedRecord:
        # ICLR proceedings record from DBLP: order_reliable=True (DBLP author
        # lists preserve publication order). structured_names=False mirrors
        # DBLP's synthesized given/family split -- the given-name audit still
        # fires because its gate is ``order_reliable``, not ``structured_names``.
        return PublishedRecord(
            doi="",
            title="Least-to-Most Prompting Enables Complex Reasoning in Large Language Models",
            authors=[{"given": "Denny", "family": "Zhou"}, {"given": "Other", "family": "Author"}],
            journal="ICLR",
            year=2023,
            order_reliable=True,
            structured_names=False,
        )

    def _arxiv_only(self) -> PublishedRecord:
        # arXiv API record (the X2 candidate-pool addition): same title+author
        # surnames, but order_reliable=False -- the arXiv listing's first author
        # is not a trustworthy ordering signal for downstream audits.
        return PublishedRecord(
            doi="10.48550/arxiv.2205.10625",
            title="Least-to-Most Prompting Enables Complex Reasoning in Large Language Models",
            authors=[{"given": "Denny", "family": "Zhou"}],
            journal="",
            year=2023,
            order_reliable=False,
            structured_names=False,
        )

    def test_order_reliable_candidate_wins_over_arxiv_only_when_tied(self, fact_checker):
        # Both score 1.0 on title+author. Without the order-reliable preference,
        # the arXiv-only record wins the confirmation tiebreak (NON_COMPARABLE
        # venue contributes no mismatch, while the DBLP record's ICLR venue
        # matches and contributes a confirmation -- but with my chosen entry
        # they tie on confirmations and the order-reliable preference settles
        # it). Assert selection picks the order-reliable DBLP record.
        cands = [(1.0, self._arxiv_only(), "arxiv"), (1.0, self._structured_dblp(), "dblp")]
        _score, rec, src = fact_checker._select_best_candidate(self._entry(), cands)
        assert src == "dblp"
        assert rec.order_reliable is True

    def test_given_name_audit_fires_after_order_reliable_selection(self, fact_checker):
        # End-to-end: the selected record is the order-reliable DBLP one, so
        # the given-name position audit runs and surfaces the lead-author
        # substitution. With the arXiv-only record selected (no fix), the audit
        # would have abstained (it requires order_reliable=True).
        cands = [(1.0, self._arxiv_only(), "arxiv"), (1.0, self._structured_dblp(), "dblp")]
        _score, rec, _src = fact_checker._select_best_candidate(self._entry(), cands)
        comps = fact_checker._compare_all_fields(self._entry(), rec)
        # The audit attaches its per-position findings to the author comparison
        # at the lead-author substitution position. The mere presence of the
        # finding is what the regression is about; the downstream status route
        # to GIVEN_NAME_SUBSTITUTION is exercised by the production bib runs.
        findings = comps["author"].given_name_findings or []
        assert any(
            f.get("position") == 0
            and f.get("variety") == "given_name_substitution"
            and f.get("entry_given") == "Shunyu"
            and f.get("record_given") == "Denny"
            for f in findings
        ), f"expected lead-author given-name substitution finding, got {findings!r}"

    def test_audit_abstains_when_only_arxiv_only_candidate_is_present(self, fact_checker):
        # Pre-fix invariant: with only an order-unreliable candidate in the
        # pool, the audit cannot fire -- this confirms the audit's gate, and
        # documents why the X2 regression surfaced when the arXiv record began
        # winning selection over a structurally-available DBLP/Crossref hit.
        cands = [(1.0, self._arxiv_only(), "arxiv")]
        _score, rec, _src = fact_checker._select_best_candidate(self._entry(), cands)
        comps = fact_checker._compare_all_fields(self._entry(), rec)
        assert rec.order_reliable is False
        assert not (comps["author"].given_name_findings or [])

    def test_arxiv_only_still_wins_when_clearly_better(self, fact_checker):
        # X2 invariant: an arXiv-only candidate that is clearly better on
        # title+author (outside the narrow order-reliable preference sub-band)
        # still wins. The sub-band is 0.02; with a 0.05 gap, the arXiv record
        # is the unambiguous best match and should be selected.
        weak_structured = PublishedRecord(
            doi="10.1/wrong",
            title="A Completely Different Paper",
            authors=[{"given": "Wrong", "family": "Author"}],
            journal="ICML",
            year=2023,
            order_reliable=True,
            structured_names=True,
        )
        cands = [(1.0, self._arxiv_only(), "arxiv"), (0.95, weak_structured, "crossref")]
        _score, rec, src = fact_checker._select_best_candidate(self._entry(), cands)
        assert src == "arxiv"
        assert rec.order_reliable is False


class TestCrossSourceAuthorFabrication:
    """FIX A: cross-source extra-author / fabrication detection. The prefix-slice
    ``symmetric_author_match`` waves through a 13-author entry that gets the
    first 5 authors right but appends fabricated trailing authors (Leak A /
    OSAKA). When two or more order-reliable candidate records agree that the
    extra entry authors are absent from the real paper, the author check is
    downgraded to MISMATCH and the status routes to AUTHOR_MISMATCH.
    """

    # Real OSAKA authors (arXiv:2003.05856 / NeurIPS 2020), 11 authors.
    REAL_AUTHORS = [
        {"given": "Massimo", "family": "Caccia"},
        {"given": "Pau", "family": "Rodriguez"},
        {"given": "Oleksiy", "family": "Ostapenko"},
        {"given": "Fabrice", "family": "Normandin"},
        {"given": "Min", "family": "Lin"},
        {"given": "Lucas", "family": "Page-Caccia"},
        {"given": "Issam Hadj", "family": "Laradji"},
        {"given": "Irina", "family": "Rish"},
        {"given": "Alexandre", "family": "Lacoste"},
        {"given": "David", "family": "Vazquez"},
        {"given": "Laurent", "family": "Charlin"},
    ]
    TITLE = "Online Fast Adaptation and Knowledge Accumulation: a New Approach to Continual Learning"

    # Entry with the documented OSAKA defect: first 5 correct, two fabricated
    # trailing surnames (Castrejon, Pineau) absent from the real paper.
    OSAKA_ENTRY_AUTHORS = (
        "Massimo Caccia and Pau Rodriguez and Oleksiy Ostapenko and "
        "Fabrice Normandin and Min Lin and Alexandre Lacoste and Laurent "
        "Charlin and Issam Laradji and Irina Rish and Alexande Lacoste and "
        "David Vazquez and Lluis Castrejon and Joelle Pineau"
    )

    def _crossref(self):
        return PublishedRecord(
            doi="10.1/x",
            title=self.TITLE,
            authors=self.REAL_AUTHORS,
            year=2020,
            structured_names=True,
            order_reliable=True,
        )

    def _dblp(self):
        return PublishedRecord(
            doi="10.1/x",
            title=self.TITLE,
            authors=self.REAL_AUTHORS,
            year=2020,
            structured_names=False,
            order_reliable=True,
        )

    def test_osaka_shape_flags_when_two_sources_agree(self, fact_checker):
        # Two order-reliable sources both lack Castrejon/Pineau -> MISMATCH.
        entry = {"title": self.TITLE, "author": self.OSAKA_ENTRY_AUTHORS, "year": "2020"}
        per_source = {"crossref": self._crossref(), "dblp": self._dblp()}
        comps = fact_checker._compare_all_fields(entry, self._crossref(), per_source_records=per_source)
        assert comps["author"].is_mismatch is True
        status = fact_checker._determine_status(0.95, comps, ["crossref", "dblp"])
        assert status == FactCheckStatus.AUTHOR_MISMATCH

    def test_single_source_does_not_flag(self, fact_checker):
        # Only one order-reliable source -> below the corroboration threshold,
        # keep the prefix-slice MATCH (a single source's record might be a stub).
        entry = {"title": self.TITLE, "author": self.OSAKA_ENTRY_AUTHORS, "year": "2020"}
        per_source = {"crossref": self._crossref()}
        comps = fact_checker._compare_all_fields(entry, self._crossref(), per_source_records=per_source)
        # No flag for fabrication: the symmetric author match still returns MATCH
        # because the prefix slice agrees and there is no cross-source signal.
        assert comps["author"].note is None or "fabricated" not in (comps["author"].note or "")

    def test_legitimate_full_match_not_flagged(self, fact_checker):
        # An entry whose authors exactly match the real list MUST verify
        # cleanly across two sources.
        entry = {
            "title": self.TITLE,
            "author": (
                "Massimo Caccia and Pau Rodriguez and Oleksiy Ostapenko and Fabrice Normandin and Min Lin "
                "and Lucas Page-Caccia and Issam Hadj Laradji and Irina Rish and Alexandre Lacoste "
                "and David Vazquez and Laurent Charlin"
            ),
            "year": "2020",
        }
        per_source = {"crossref": self._crossref(), "dblp": self._dblp()}
        comps = fact_checker._compare_all_fields(entry, self._crossref(), per_source_records=per_source)
        assert comps["author"].is_mismatch is False
        assert comps["author"].is_confirmed is True

    def test_full_author_best_match_from_order_unreliable_source_not_flagged(self, fact_checker):
        # Regression (joshi2026guardians, UAI 2026): the full-author record came
        # from an order-UNRELIABLE source (arXiv), while the order-reliable
        # sources (Crossref/OpenAlex) returned an incomplete stub for this very
        # recent paper (only the first author indexed). Every entry author
        # appears in the best-matched record, so none is fabricated -- the
        # positive-presence veto must keep the author check a clean MATCH rather
        # than flagging the five stub-absent surnames.
        title = "Who Guards the Guardians? The Challenges of Evaluating Identifiability of Learned Representations"
        full_authors = [
            {"given": "Shruti", "family": "Joshi"},
            {"given": "Théo", "family": "Saulus"},
            {"given": "Wieland", "family": "Brendel"},
            {"given": "Philippe", "family": "Brouillard"},
            {"given": "Dhanya", "family": "Sridhar"},
            {"given": "Patrik", "family": "Reizinger"},
        ]
        # Best match: full author list, but arXiv is order-unreliable.
        arxiv_rec = PublishedRecord(
            doi="",
            title=title,
            authors=full_authors,
            year=2026,
            structured_names=True,
            order_reliable=False,
        )
        # Two order-reliable stubs: a brand-new paper with only the lead author
        # indexed in Crossref/OpenAlex.
        stub_authors = [{"given": "Shruti", "family": "Joshi"}]
        crossref_stub = PublishedRecord(
            doi="", title=title, authors=stub_authors, year=2026, structured_names=True, order_reliable=True
        )
        openalex_stub = PublishedRecord(
            doi="", title=title, authors=stub_authors, year=2026, structured_names=True, order_reliable=True
        )
        entry = {
            "title": title,
            "author": (
                "Joshi, Shruti and Saulus, Th{\\'e}o and Brendel, Wieland and "
                "Brouillard, Philippe and Sridhar, Dhanya and Reizinger, Patrik"
            ),
            "year": "2026",
        }
        per_source = {"crossref": crossref_stub, "openalex": openalex_stub}
        comps = fact_checker._compare_all_fields(entry, arxiv_rec, per_source_records=per_source)
        assert comps["author"].is_mismatch is False
        assert comps["author"].is_confirmed is True
        assert "fabricated" not in (comps["author"].note or "")

    def test_entry_shorter_than_candidate_not_flagged(self, fact_checker):
        # A shorter entry (e.g. only the first few authors) is consistent with
        # the candidate -- not fabrication. symmetric_author_match returns
        # PARTIAL (leading prefix), so FIX A's MATCH-gated check stays off.
        entry = {
            "title": self.TITLE,
            "author": "Massimo Caccia and Pau Rodriguez and Oleksiy Ostapenko",
            "year": "2020",
        }
        per_source = {"crossref": self._crossref(), "dblp": self._dblp()}
        comps = fact_checker._compare_all_fields(entry, self._crossref(), per_source_records=per_source)
        # Not a MISMATCH -- either MATCH (sentinel-aware leading head) or PARTIAL.
        assert comps["author"].is_mismatch is False

    def test_ordinalclip_shape_short_candidate_with_fabricated_trailing(self, fact_checker):
        # Leak C / OrdinalCLIP shape: the candidate records are SHORTER than the
        # entry (e.g. an API stub with only 3 of the real authors), and the
        # entry's trailing surnames are fabricated. ``symmetric_author_match``
        # returns PARTIAL (the candidate is a leading prefix of the entry), so
        # the original code routed to UNCONFIRMED -- which buries a real
        # fabrication signal. With two order-reliable sources agreeing on the
        # short list, the trailing entry surnames are absent from BOTH unions,
        # so FIX A escalates from PARTIAL to MISMATCH.
        short_authors = [
            {"given": "Shaoyuan", "family": "Li"},
            {"given": "Hua", "family": "Liu"},
            {"given": "Wei", "family": "Hu"},
        ]
        cr_rec = PublishedRecord(
            doi="10.1/x",
            title="OrdinalCLIP",
            authors=short_authors,
            year=2022,
            structured_names=True,
            order_reliable=True,
        )
        dblp_rec = PublishedRecord(
            doi="10.1/x",
            title="OrdinalCLIP",
            authors=short_authors,
            year=2022,
            structured_names=False,
            order_reliable=True,
        )
        entry = {
            "title": "OrdinalCLIP",
            "author": "Shaoyuan Li and Hua Liu and Wei Hu and Phantom One and Phantom Two and Phantom Three",
            "year": "2022",
        }
        per_source = {"crossref": cr_rec, "dblp": dblp_rec}
        comps = fact_checker._compare_all_fields(entry, cr_rec, per_source_records=per_source)
        assert comps["author"].is_mismatch is True
        status = fact_checker._determine_status(0.95, comps, ["crossref", "dblp"])
        assert status == FactCheckStatus.AUTHOR_MISMATCH

    def test_sentinel_entry_not_flagged(self, fact_checker):
        # "and others" on the entry side suppresses the fabrication check: the
        # citation is explicitly truncated, so a trailing surname's absence
        # cannot be read as fabrication.
        entry = {
            "title": self.TITLE,
            "author": (
                "Massimo Caccia and Pau Rodriguez and Oleksiy Ostapenko and Fabrice Normandin "
                "and Lluis Castrejon and Joelle Pineau and others"
            ),
            "year": "2020",
        }
        per_source = {"crossref": self._crossref(), "dblp": self._dblp()}
        comps = fact_checker._compare_all_fields(entry, self._crossref(), per_source_records=per_source)
        # No fabrication note: sentinel suppresses the check.
        assert "fabricated" not in (comps["author"].note or "")


class TestFirstAuthorGivenNameSubstitution:
    """FIX B: the existing ``given_name_position_audit`` previously skipped the
    first author when Crossref returned a ``literal`` (no given/family split)
    for the lead author. Leak B (Least-to-Most: "Shunyu Zhou" cited where the
    real lead is "Denny Zhou") slipped through because the synthesized-empty
    family produced an empty surname key -- the audit's surname-key pairing
    silently dropped position 0. The fix routes empty-family records through
    ``last_name_from_person`` on the reconstructed name, symmetric with
    ``PublishedRecord.surname_keys``.
    """

    TITLE = "Least-to-Most Prompting Enables Complex Reasoning"

    def test_literal_lead_substitution_flags(self, fact_checker):
        # Crossref literal-author record: ``given`` carries the full flat name,
        # ``family`` is empty. The audit must still pair position 0 and detect
        # SUBSTITUTION.
        record = PublishedRecord(
            doi="10.1/x",
            title=self.TITLE,
            authors=[
                {"given": "Denny Zhou", "family": ""},
                {"given": "Nathanael", "family": "Schaerli"},
            ],
            year=2023,
            structured_names=True,
            order_reliable=True,
        )
        entry = {"title": self.TITLE, "author": "Shunyu Zhou and Nathanael Schaerli", "year": "2023"}
        comps = fact_checker._compare_all_fields(entry, record)
        assert comps["author"].is_mismatch is True
        status = fact_checker._determine_status(0.95, comps, ["crossref"])
        assert status == FactCheckStatus.GIVEN_NAME_SUBSTITUTION

    def test_literal_lead_correct_given_verifies(self, fact_checker):
        # The same record/path with the CORRECT lead author must still verify
        # (no false positive on the empty-family code path).
        record = PublishedRecord(
            doi="10.1/x",
            title=self.TITLE,
            authors=[
                {"given": "Denny Zhou", "family": ""},
                {"given": "Nathanael", "family": "Schaerli"},
            ],
            year=2023,
            structured_names=True,
            order_reliable=True,
        )
        entry = {"title": self.TITLE, "author": "Denny Zhou and Nathanael Schaerli", "year": "2023"}
        comps = fact_checker._compare_all_fields(entry, record)
        assert comps["author"].is_mismatch is False

    def test_lead_substitution_when_entry_duplicates_surname(self, fact_checker):
        # Refinement: the real Least-to-Most leak had the entry citing
        # "Shunyu Zhou" as lead AND re-listing the canonical lead "Denny Zhou"
        # at the tail. The OLD repeated-surname guard skipped the lead-position
        # audit because the surname 'zhou' appeared TWICE in the entry, even
        # though the record had it only ONCE (at position 0). The refined
        # guard requires BOTH sides to repeat before declaring positional
        # ambiguity, so the lead audit now fires for this shape.
        record = PublishedRecord(
            doi="10.1/x",
            title=self.TITLE,
            authors=[
                {"given": "Denny", "family": "Zhou"},
                {"given": "Nathanael", "family": "Schaerli"},
            ],
            year=2023,
            structured_names=True,
            order_reliable=True,
        )
        entry = {
            "title": self.TITLE,
            "author": "Shunyu Zhou and Nathanael Schaerli and Denny Zhou",
            "year": "2023",
        }
        comps = fact_checker._compare_all_fields(entry, record)
        assert comps["author"].is_mismatch is True
        status = fact_checker._determine_status(0.95, comps, ["crossref"])
        assert status == FactCheckStatus.GIVEN_NAME_SUBSTITUTION

    def test_lead_both_sides_repeat_with_benign_given_match_abstains(self, fact_checker):
        # When BOTH entry AND record have the lead surname repeated, there is
        # genuine positional ambiguity. If the entry's lead given matches the
        # record's position-0 same-surname given via a benign class (here:
        # initial-compatible "Y." vs "Yang"), the audit abstains rather than
        # escalating. Entry order matches record order, so the separate
        # same-surname swap detector also stays quiet.
        record = PublishedRecord(
            doi="10.1/x",
            title=self.TITLE,
            authors=[
                {"given": "Yang", "family": "Song"},
                {"given": "Jiaming", "family": "Song"},
            ],
            year=2023,
            structured_names=True,
            order_reliable=True,
        )
        # Entry: lead "Y. Song" (initial-compatible with record's "Yang Song")
        # then "Jiaming Song" -- order preserved.
        entry = {
            "title": self.TITLE,
            "author": "Y. Song and Jiaming Song",
            "year": "2023",
        }
        comps = fact_checker._compare_all_fields(entry, record)
        # "Y." is an INITIAL_COMPATIBLE benign match for "Yang" -> not a mismatch.
        assert comps["author"].is_mismatch is False

    def test_lead_both_sides_repeat_with_no_benign_match_escalates(self, fact_checker):
        # Same both-sides-repeat shape, but the entry's lead given matches
        # NEITHER of the record's same-surname givens via any benign class
        # (EXACT/INITIAL/ABBREVIATION/etc.). That is a real substitution
        # whichever record author the entry meant -> escalate.
        record = PublishedRecord(
            doi="10.1/x",
            title=self.TITLE,
            authors=[
                {"given": "Jiaming", "family": "Song"},
                {"given": "Yang", "family": "Song"},
            ],
            year=2023,
            structured_names=True,
            order_reliable=True,
        )
        # Entry leads with "Phantom Song" -- not Jiaming, not Yang.
        entry = {
            "title": self.TITLE,
            "author": "Phantom Song and Jiaming Song and Yang Song",
            "year": "2023",
        }
        comps = fact_checker._compare_all_fields(entry, record)
        assert comps["author"].is_mismatch is True
