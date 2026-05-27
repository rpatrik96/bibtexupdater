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

    def test_verify_arxiv_match_rejects_title_mismatch(self, resolver):
        """Regression (onebench/agrawal): Stage 1/1b trust ``detection.arxiv_id``
        and assign confidence 1.0. If the cited arXiv ID is wrong, the record it
        maps to is an unrelated paper; _verify_arxiv_match must reject it so the
        cascade falls through to title-based resolution instead of silently
        rewriting the entry into that other paper.
        """
        entry = {
            "title": "ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities",
            "author": "Ghosh, Adhiraj and Dziadzio, Sebastian and Prabhu, Ameya",
        }
        # The record the wrong arXiv ID (2412.06745) actually resolves to.
        wrong_paper = PublishedRecord(
            doi="10.1000/robotron",
            title="RoboTron-Drive: All-in-One Large Multimodal Model for Autonomous Driving",
            journal="Some Journal",
            year=2024,
            authors=[{"given": "Zhijian", "family": "Huang"}, {"given": "Chengjian", "family": "Feng"}],
            type="journal-article",
        )
        title_norm = normalize_title_for_match(entry["title"])
        assert resolver._verify_arxiv_match(wrong_paper, entry, title_norm) is None

    def test_verify_arxiv_match_accepts_title_match(self, resolver):
        """A record whose title/author match the entry passes the gate unchanged."""
        entry = {
            "title": "ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities",
            "author": "Ghosh, Adhiraj and Dziadzio, Sebastian and Prabhu, Ameya",
        }
        right_paper = PublishedRecord(
            doi="10.18653/v1/2025.acl-long.1560",
            title="ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities",
            journal="ACL",
            year=2025,
            authors=[
                {"given": "Adhiraj", "family": "Ghosh"},
                {"given": "Sebastian", "family": "Dziadzio"},
                {"given": "Ameya", "family": "Prabhu"},
            ],
            type="proceedings-article",
        )
        title_norm = normalize_title_for_match(entry["title"])
        assert resolver._verify_arxiv_match(right_paper, entry, title_norm) is right_paper

    def test_verify_arxiv_match_passthrough_without_title(self, resolver):
        """With no entry title to verify against, the direct ID lookup is trusted."""
        rec = PublishedRecord(doi="10.1/x", title="Whatever Paper", year=2024, authors=[])
        assert resolver._verify_arxiv_match(rec, {"title": ""}, "") is rec

    def test_verify_arxiv_match_none_stays_none(self, resolver):
        """A miss (None) is propagated unchanged."""
        assert resolver._verify_arxiv_match(None, {"title": "anything"}, "anything") is None

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


# Exact failing S2 arXiv payload for arXiv:2406.14302 ("Identifiable
# Exchangeable Mechanisms ..."): S2 keeps the arXiv preprint DOI + 2024 year
# while reporting the *published* ICLR venue via publicationVenue. The record
# is internally inconsistent and must NOT be used to upgrade the entry.
IEM_S2_ARXIV_PAYLOAD = {
    "paperId": "deadbeef",
    "title": "Identifiable Exchangeable Mechanisms for Causal Structure and " "Representation Learning",
    "year": 2024,
    "publicationDate": "2024-06-20",
    "externalIds": {"ArXiv": "2406.14302", "DOI": "10.48550/arXiv.2406.14302"},
    "venue": "International Conference on Learning Representations",
    "publicationTypes": ["JournalArticle"],
    "publicationVenue": {"name": "International Conference on Learning Representations"},
    "journal": {"name": "ArXiv"},
    "url": "https://www.semanticscholar.org/paper/deadbeef",
    "authors": [
        {"name": "Patrik Reizinger"},
        {"name": "Siyuan Guo"},
    ],
}

# A genuinely published S2 arXiv payload: real publisher DOI + a real
# (non-preprint) journal name. This must still be accepted.
GENUINE_S2_ARXIV_PAYLOAD = {
    "paperId": "cafef00d",
    "title": "A Genuinely Published Paper",
    "year": 2023,
    "publicationDate": "2023-05-01",
    "externalIds": {"ArXiv": "2301.00001", "DOI": "10.1162/neco_a_01567"},
    "venue": "Neural Computation",
    "publicationTypes": ["JournalArticle"],
    "publicationVenue": {"name": "Neural Computation"},
    "journal": {"name": "Neural Computation"},
    "url": "https://www.semanticscholar.org/paper/cafef00d",
    "authors": [
        {"name": "Jane Doe"},
        {"name": "John Smith"},
    ],
}


# A bioRxiv preprint that S2 tagged with a published venue but left under a
# bioRxiv DOI, with NO journal object at all (so only the DOI arm of the guard
# can catch it). Mirrors Detector.detect, which treats 10.1101 as a preprint
# DOI just like 10.48550/arxiv.
BIORXIV_S2_ARXIV_PAYLOAD = {
    "paperId": "b10b10b1",
    "title": "A bioRxiv Preprint Tagged With a Published Venue",
    "year": 2024,
    "publicationDate": "2024-01-15",
    "externalIds": {"ArXiv": "2401.99999", "DOI": "10.1101/2024.01.15.575678"},
    "venue": "Nature Methods",
    "publicationTypes": ["JournalArticle"],
    "publicationVenue": {"name": "Nature Methods"},
    "url": "https://www.semanticscholar.org/paper/b10b10b1",
    "authors": [
        {"name": "Patrik Reizinger"},
        {"name": "Siyuan Guo"},
    ],
}


def _resp(payload, status_code=200):
    """Build a MagicMock HTTP response mirroring tests/test_cascade_sources.py."""
    return MagicMock(status_code=status_code, json=lambda: payload)


class TestS2FromArxivPreprintGuard:
    """S2 arXiv builders must reject records that only tag a preprint with a
    published venue while still carrying the arXiv DOI/year (the resolution
    cascade should then fall through to DBLP for the real published record).
    """

    @pytest.fixture
    def resolver(self, logger):
        # MagicMock http (not FakeHttpClient, which raises on _request).
        return Resolver(http=MagicMock(), logger=logger, scholarly_client=None)

    # --- 1. Reject case: sync Resolver.s2_from_arxiv ---
    def test_s2_from_arxiv_rejects_preprint_with_published_venue(self, resolver):
        resolver.http._request.return_value = _resp(IEM_S2_ARXIV_PAYLOAD)
        assert resolver.s2_from_arxiv("2406.14302") is None

    # --- 1b. Reject case: bioRxiv-DOI-only payload (only the DOI arm can
    #          catch it; no journal.name). Mirrors Detector.detect's 10.1101. ---
    def test_s2_from_arxiv_rejects_biorxiv_doi_with_published_venue(self, resolver):
        resolver.http._request.return_value = _resp(BIORXIV_S2_ARXIV_PAYLOAD)
        assert resolver.s2_from_arxiv("2401.99999") is None

    def test_async_s2_from_arxiv_rejects_biorxiv_doi_with_published_venue(self, logger):
        async def _run():
            http = MagicMock()

            async def _get(*args, **kwargs):
                return _resp(BIORXIV_S2_ARXIV_PAYLOAD)

            http.get = _get
            ares = AsyncResolver(http=http, logger=logger)
            return await ares.s2_from_arxiv("2401.99999")

        assert asyncio.run(_run()) is None

    # --- 2. Non-regression: genuine published S2 record still accepted ---
    def test_s2_from_arxiv_accepts_genuine_published_record(self, resolver):
        resolver.http._request.return_value = _resp(GENUINE_S2_ARXIV_PAYLOAD)
        rec = resolver.s2_from_arxiv("2301.00001")
        assert rec is not None
        assert rec.doi == "10.1162/neco_a_01567"
        assert rec.journal == "Neural Computation"
        assert rec.year == 2023

    # --- 3a. Reject case: s2_batch_lookup ---
    def test_s2_batch_lookup_rejects_preprint_with_published_venue(self, resolver):
        resolver.http._request.return_value = _resp([IEM_S2_ARXIV_PAYLOAD])
        results = resolver.s2_batch_lookup(["arXiv:2406.14302"])
        assert results["arXiv:2406.14302"] is None

    def test_s2_batch_lookup_accepts_genuine_published_record(self, resolver):
        resolver.http._request.return_value = _resp([GENUINE_S2_ARXIV_PAYLOAD])
        results = resolver.s2_batch_lookup(["arXiv:2301.00001"])
        rec = results["arXiv:2301.00001"]
        assert rec is not None
        assert rec.doi == "10.1162/neco_a_01567"
        assert rec.journal == "Neural Computation"

    # --- 3b. Reject case: AsyncResolver.s2_from_arxiv ---
    def test_async_s2_from_arxiv_rejects_preprint_with_published_venue(self, logger):
        async def _run():
            http = MagicMock()

            async def _get(*args, **kwargs):
                return _resp(IEM_S2_ARXIV_PAYLOAD)

            http.get = _get
            ares = AsyncResolver(http=http, logger=logger)
            return await ares.s2_from_arxiv("2406.14302")

        assert asyncio.run(_run()) is None

    def test_async_s2_from_arxiv_accepts_genuine_published_record(self, logger):
        async def _run():
            http = MagicMock()

            async def _get(*args, **kwargs):
                return _resp(GENUINE_S2_ARXIV_PAYLOAD)

            http.get = _get
            ares = AsyncResolver(http=http, logger=logger)
            return await ares.s2_from_arxiv("2301.00001")

        rec = asyncio.run(_run())
        assert rec is not None
        assert rec.doi == "10.1162/neco_a_01567"

    # --- 4a. Stage-1 short-circuit no longer fires for the reject payload ---
    def test_stage1_does_not_short_circuit_on_preprint_venue(self, resolver):
        """With the reject payload from S2 and no arXiv->Crossref DOI, Stage 1
        must return (None, ...) so the cascade continues to DBLP."""

        def _dispatch(method, url, *args, **kwargs):
            if "semanticscholar" in url or "/paper/arXiv:" in url:
                return _resp(IEM_S2_ARXIV_PAYLOAD)
            # arXiv API: no published <arxiv:doi> -> candidate_doi is None
            return MagicMock(status_code=200, text="<feed></feed>")

        resolver.http._request.side_effect = _dispatch
        detection = PreprintDetection(is_preprint=True, reason="eprint arXiv", arxiv_id="2406.14302", doi=None)
        rec, candidate_doi = resolver._stage1_direct_lookup(detection)
        assert rec is None

    # --- 4b. Cascade falls through to DBLP -> correct ICLR-2025 record ---
    def test_cascade_falls_through_to_dblp_iclr_2025(self, resolver):
        """End-to-end: S2 yields the reject payload, arXiv has no published
        DOI, OpenAlex misses, but DBLP returns the real ICLR-2025 record.
        The resolved record must have year=2025, an ICLR venue, and NO
        10.48550/arxiv DOI."""

        dblp_hit = {
            "info": {
                "title": "Identifiable Exchangeable Mechanisms for Causal " "Structure and Representation Learning",
                "authors": {
                    "author": [
                        {"text": "Patrik Reizinger"},
                        {"text": "Siyuan Guo"},
                    ]
                },
                "venue": "International Conference on Learning Representations",
                "year": "2025",
                "type": "Conference and Workshop Papers",
                "ee": "https://openreview.net/forum?id=k03mB41vyM",
            }
        }

        def _dispatch(method, url, *args, **kwargs):
            if "/paper/arXiv:" in url or "semanticscholar" in url:
                return _resp(IEM_S2_ARXIV_PAYLOAD)
            if "dblp.org" in url:
                return _resp({"result": {"hits": {"hit": [dblp_hit]}}})
            if "export.arxiv.org" in url:
                return MagicMock(status_code=200, text="<feed></feed>")
            # OpenAlex / Crossref / everything else: miss
            return MagicMock(status_code=404, json=lambda: {})

        resolver.http._request.side_effect = _dispatch
        entry = {
            "ID": "iem2024",
            "ENTRYTYPE": "article",
            "title": "Identifiable Exchangeable Mechanisms for Causal " "Structure and Representation Learning",
            "author": "Reizinger, Patrik and Guo, Siyuan",
            "year": "2024",
        }
        detection = PreprintDetection(is_preprint=True, reason="eprint arXiv", arxiv_id="2406.14302", doi=None)
        rec = resolver.resolve(entry, detection)
        assert rec is not None
        assert rec.year == 2025
        assert "learning representations" in (rec.journal or "").lower()
        assert "10.48550/arxiv" not in (rec.doi or "").lower()
