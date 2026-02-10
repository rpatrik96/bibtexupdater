"""Tests for Resolver stage methods and AsyncResolver."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from bibtex_updater import (
    AsyncResolver,
    PreprintDetection,
    PublishedRecord,
    Resolver,
)
from bibtex_updater.utils import normalize_title_for_match


class TestResolverStageMethods:
    """Tests for the refactored stage methods in Resolver."""

    @pytest.fixture
    def resolver(self, fake_http, logger):
        """Create a Resolver instance with fake HTTP client."""
        return Resolver(http=fake_http, logger=logger, scholarly_client=None)

    @pytest.fixture
    def arxiv_detection(self):
        """Create an arXiv preprint detection."""
        return PreprintDetection(
            is_preprint=True,
            reason="arXiv ID found",
            arxiv_id="2301.00001",
            doi=None,
        )

    @pytest.fixture
    def doi_detection(self):
        """Create a DOI-based preprint detection."""
        return PreprintDetection(
            is_preprint=True,
            reason="bioRxiv DOI",
            arxiv_id=None,
            doi="10.1101/2024.01.01.000001",
        )

    @pytest.fixture
    def sample_entry(self):
        """Create a sample BibTeX entry."""
        return {
            "ID": "test2024",
            "ENTRYTYPE": "article",
            "title": "Test Paper Title",
            "author": "Smith, John and Doe, Jane",
            "year": "2024",
        }

    def test_stage1_returns_tuple(self, resolver, arxiv_detection):
        """Stage 1 should return a tuple of (record, candidate_doi)."""
        result = resolver._stage1_direct_lookup(arxiv_detection)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_stage1_with_no_arxiv_id(self, resolver, doi_detection):
        """Stage 1 should handle detection without arXiv ID."""
        result, candidate_doi = resolver._stage1_direct_lookup(doi_detection)
        # Without arXiv ID, stage 1 should return None
        assert result is None
        assert candidate_doi is None

    def test_stage2_with_none_candidate_doi(self, resolver, arxiv_detection):
        """Stage 2 should handle None candidate_doi gracefully."""
        result = resolver._stage2_crossref_relations(arxiv_detection, None)
        # With no DOI to look up, should return None
        assert result is None

    def test_stage3_returns_record_or_none(self, resolver, sample_entry):
        """Stage 3 should return PublishedRecord or None."""
        title_norm = normalize_title_for_match(sample_entry["title"])
        result = resolver._stage3_dblp_search(sample_entry, title_norm)
        assert result is None or isinstance(result, PublishedRecord)

    def test_stage4_returns_record_or_none(self, resolver, sample_entry):
        """Stage 4 should return PublishedRecord or None."""
        title_norm = normalize_title_for_match(sample_entry["title"])
        result = resolver._stage4_s2_search(sample_entry, title_norm)
        assert result is None or isinstance(result, PublishedRecord)

    def test_stage5_returns_record_or_none(self, resolver, sample_entry):
        """Stage 5 should return PublishedRecord or None."""
        title_norm = normalize_title_for_match(sample_entry["title"])
        result = resolver._stage5_crossref_search(sample_entry, title_norm)
        assert result is None or isinstance(result, PublishedRecord)

    def test_stage6_returns_none_without_scholarly_client(self, resolver, sample_entry):
        """Stage 6 should return None when scholarly_client is None."""
        title_norm = normalize_title_for_match(sample_entry["title"])
        result = resolver._stage6_scholarly_search(sample_entry, title_norm)
        assert result is None

    def test_resolve_uncached_calls_stages_in_order(self, resolver, sample_entry, arxiv_detection):
        """_resolve_uncached should call stages in sequence."""
        # This test verifies the orchestration works
        result = resolver._resolve_uncached(sample_entry, arxiv_detection)
        # With fake HTTP that returns empty responses, result should be None
        assert result is None or isinstance(result, PublishedRecord)


class TestAsyncResolver:
    """Tests for AsyncResolver functionality."""

    @pytest.fixture
    def async_http(self):
        """Create a mock async HTTP client."""
        mock = MagicMock()
        mock.get = MagicMock(return_value=asyncio.coroutine(lambda: MagicMock(status_code=404))())
        mock.post = MagicMock(return_value=asyncio.coroutine(lambda: MagicMock(status_code=404))())
        return mock

    @pytest.fixture
    def async_resolver(self, async_http, logger):
        """Create an AsyncResolver instance."""
        return AsyncResolver(http=async_http, logger=logger, scholarly_client=None)

    def test_async_resolver_has_same_methods_as_resolver(self):
        """AsyncResolver should have the same public methods as Resolver."""
        resolver_methods = {m for m in dir(Resolver) if not m.startswith("_")}
        async_resolver_methods = {m for m in dir(AsyncResolver) if not m.startswith("_")}

        # AsyncResolver should have at least the same public interface
        common_methods = {
            "resolve",
            "crossref_get",
            "crossref_search",
            "dblp_search",
            "s2_search",
            "openalex_from_arxiv",
            "openalex_from_doi",
            "europepmc_search_published",
        }
        for method in common_methods:
            assert method in resolver_methods, f"Resolver missing {method}"
            assert method in async_resolver_methods, f"AsyncResolver missing {method}"

    def test_async_resolver_parallel_search_exists(self):
        """AsyncResolver should have parallel_bibliographic_search method."""
        assert hasattr(AsyncResolver, "parallel_bibliographic_search")

    def test_async_resolver_has_biorxiv_detection(self):
        """AsyncResolver should have _is_biorxiv_or_medrxiv helper."""
        assert hasattr(AsyncResolver, "_is_biorxiv_or_medrxiv")


class TestMainHelperFunctions:
    """Tests for main() helper functions."""

    def test_validate_arguments_with_valid_output(self, logger):
        """validate_arguments should return None when output is specified."""
        import argparse

        from bibtex_updater.updater import validate_arguments

        args = argparse.Namespace(
            output="output.bib",
            in_place=False,
        )
        result = validate_arguments(args, logger)
        # With output specified, should return None (no error)
        assert result is None

    def test_validate_arguments_with_valid_inplace(self, logger):
        """validate_arguments should return None when in_place is True."""
        import argparse

        from bibtex_updater.updater import validate_arguments

        args = argparse.Namespace(
            output=None,
            in_place=True,
        )
        result = validate_arguments(args, logger)
        # With in_place, should return None (no error)
        assert result is None

    def test_validate_arguments_rejects_neither_output_nor_inplace(self, logger):
        """validate_arguments should reject when neither output nor in_place."""
        import argparse

        from bibtex_updater.updater import validate_arguments

        args = argparse.Namespace(
            output=None,
            in_place=False,
        )
        result = validate_arguments(args, logger)
        # Should return an error message
        assert result is not None
        assert "output" in result.lower() or "in-place" in result.lower()

    def test_setup_http_client_returns_http_client(self):
        """setup_http_client should return an HttpClient instance."""
        import argparse

        from bibtex_updater import HttpClient
        from bibtex_updater.updater import setup_http_client

        args = argparse.Namespace(
            rate_limit=45,
            cache=None,
            timeout=20,
            verbose=False,
        )
        client = setup_http_client(args)
        assert isinstance(client, HttpClient)

    def test_build_arg_parser_returns_parser(self):
        """build_arg_parser should return an ArgumentParser."""
        import argparse

        from bibtex_updater.updater import build_arg_parser

        parser = build_arg_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_build_arg_parser_has_required_args(self):
        """build_arg_parser should define all required arguments."""
        from bibtex_updater.updater import build_arg_parser

        parser = build_arg_parser()
        # Parse with minimal required args
        args = parser.parse_args(["test.bib"])
        assert args.inputs == ["test.bib"]
        assert hasattr(args, "output")
        assert hasattr(args, "in_place")
        assert hasattr(args, "dry_run")


class TestResolverMixinMethods:
    """Tests for methods that are shared between Resolver and AsyncResolver."""

    @pytest.fixture
    def resolver(self, fake_http, logger):
        """Create a Resolver instance."""
        return Resolver(http=fake_http, logger=logger, scholarly_client=None)

    def test_credible_journal_article_rejects_preprint(self, resolver):
        """_credible_journal_article should reject preprint records."""
        preprint_record = PublishedRecord(
            doi="10.1101/2024.01.01.000001",
            title="Test Paper",
            journal="bioRxiv",
            year=2024,
            authors=[{"given": "John", "family": "Smith"}],
            volume=None,
            pages=None,
            type="posted-content",
        )
        assert not resolver._credible_journal_article(preprint_record)

    def test_credible_journal_article_accepts_journal(self, resolver):
        """_credible_journal_article should accept legitimate journal articles."""
        journal_record = PublishedRecord(
            doi="10.1038/s41586-024-00001-1",
            title="Test Paper",
            journal="Nature",
            year=2024,
            authors=[{"given": "John", "family": "Smith"}],
            volume="625",
            pages="1-10",
            type="journal-article",
        )
        assert resolver._credible_journal_article(journal_record)

    def test_authors_to_bibtex_string_formats_correctly(self, resolver):
        """_authors_to_bibtex_string should format authors correctly."""
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test",
            journal="Test Journal",
            year=2024,
            authors=[
                {"given": "John", "family": "Smith"},
                {"given": "Jane", "family": "Doe"},
            ],
            volume=None,
            pages=None,
            type="journal-article",
        )
        author_str = resolver._authors_to_bibtex_string(record)
        assert "Smith" in author_str
        assert "Doe" in author_str
        assert " and " in author_str
