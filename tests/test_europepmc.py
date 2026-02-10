"""Tests for Europe PMC integration (Stage 1c)."""

from __future__ import annotations

import pytest

from bibtex_updater import PreprintDetection, PublishedRecord, Resolver
from bibtex_updater.utils import europepmc_result_to_record, normalize_title_for_match

# --- europepmc_result_to_record converter tests ---


class TestEuropePMCResultToRecord:
    """Tests for the europepmc_result_to_record converter function."""

    def test_valid_published_result(self):
        """Should convert a published article from Europe PMC."""
        result = {
            "doi": "10.1038/s41586-024-00001-1",
            "title": "Novel Gene Discovery in Human Cells",
            "pubYear": "2024",
            "journalTitle": "Nature",
            "authorString": "Jones A, Brown B, Smith C",
            "source": "MED",
            "journalVolume": "625",
            "journalIssueId": "7995",
            "pageInfo": "100-110",
        }
        rec = europepmc_result_to_record(result)
        assert rec is not None
        assert rec.doi == "10.1038/s41586-024-00001-1"
        assert rec.title == "Novel Gene Discovery in Human Cells"
        assert rec.year == 2024
        assert rec.journal == "Nature"
        assert len(rec.authors) >= 1

    def test_rejects_preprint_source(self):
        """Should reject results from preprint sources (PPR)."""
        result = {
            "doi": "10.1101/2024.01.01.123456",
            "title": "A Preprint Paper",
            "pubYear": "2024",
            "journalTitle": "bioRxiv",
            "authorString": "Doe J",
            "source": "PPR",
        }
        rec = europepmc_result_to_record(result)
        assert rec is None

    def test_handles_no_doi(self):
        """Should return record with empty DOI when DOI is missing."""
        result = {
            "title": "Paper Without DOI",
            "pubYear": "2024",
            "journalTitle": "Some Journal",
            "authorString": "Smith J",
            "source": "MED",
        }
        rec = europepmc_result_to_record(result)
        # Converter returns a record even without DOI; credibility check filters later
        assert rec is not None
        assert rec.doi == ""

    def test_empty_result(self):
        """Should handle empty result objects."""
        rec = europepmc_result_to_record({})
        assert rec is None

    def test_author_parsing_standard_format(self):
        """Should parse 'FamilyName Initials' author format."""
        result = {
            "doi": "10.1234/test",
            "title": "Test Paper",
            "pubYear": "2024",
            "journalTitle": "Test Journal",
            "authorString": "Smith JA, Jones B, Brown CD",
            "source": "MED",
            "journalVolume": "1",
        }
        rec = europepmc_result_to_record(result)
        assert rec is not None
        assert len(rec.authors) == 3
        assert rec.authors[0]["family"] == "Smith"
        assert rec.authors[1]["family"] == "Jones"

    def test_handles_missing_fields(self):
        """Should handle missing optional fields gracefully."""
        result = {
            "doi": "10.1234/minimal",
            "title": "Minimal Paper",
            "pubYear": "2024",
            "source": "MED",
        }
        rec = europepmc_result_to_record(result)
        # May return None or a record depending on implementation
        # (no journal info means not credible, but converter should still work)
        assert rec is None or isinstance(rec, PublishedRecord)


# --- Resolver Stage 1c tests ---


class TestStage1cEuropePMC:
    """Tests for the Europe PMC resolution stage."""

    @pytest.fixture
    def resolver(self, fake_http, logger):
        return Resolver(http=fake_http, logger=logger, scholarly_client=None)

    @pytest.fixture
    def biorxiv_entry(self):
        return {
            "ID": "biorxiv2024",
            "ENTRYTYPE": "article",
            "title": "Novel Gene Discovery",
            "author": "Jones, Alice and Brown, Bob",
            "journal": "bioRxiv",
            "doi": "10.1101/2024.01.01.123456",
            "year": "2024",
        }

    @pytest.fixture
    def biorxiv_detection(self):
        return PreprintDetection(
            is_preprint=True,
            reason="bioRxiv DOI",
            arxiv_id=None,
            doi="10.1101/2024.01.01.123456",
        )

    @pytest.fixture
    def arxiv_entry(self):
        return {
            "ID": "arxiv2024",
            "ENTRYTYPE": "article",
            "title": "Deep Learning Paper",
            "author": "Smith, John",
            "journal": "arXiv preprint arXiv:2301.00001",
            "year": "2024",
        }

    @pytest.fixture
    def arxiv_detection(self):
        return PreprintDetection(
            is_preprint=True,
            reason="arXiv",
            arxiv_id="2301.00001",
            doi=None,
        )

    def test_is_biorxiv_doi_prefix(self, resolver, biorxiv_entry, biorxiv_detection):
        """Should detect bioRxiv by 10.1101/ DOI prefix."""
        assert resolver._is_biorxiv_or_medrxiv(biorxiv_entry, biorxiv_detection) is True

    def test_is_biorxiv_journal_field(self, resolver):
        """Should detect bioRxiv from journal field."""
        entry = {"journal": "bioRxiv"}
        detection = PreprintDetection(is_preprint=True, reason="test", arxiv_id=None, doi=None)
        assert resolver._is_biorxiv_or_medrxiv(entry, detection) is True

    def test_is_medrxiv_journal_field(self, resolver):
        """Should detect medRxiv from journal field."""
        entry = {"journal": "medRxiv"}
        detection = PreprintDetection(is_preprint=True, reason="test", arxiv_id=None, doi=None)
        assert resolver._is_biorxiv_or_medrxiv(entry, detection) is True

    def test_not_biorxiv_for_arxiv(self, resolver, arxiv_entry, arxiv_detection):
        """Should NOT detect arXiv entries as bioRxiv."""
        assert resolver._is_biorxiv_or_medrxiv(arxiv_entry, arxiv_detection) is False

    def test_stage1c_skips_non_biorxiv(self, resolver, arxiv_entry, arxiv_detection):
        """Stage 1c should skip non-bioRxiv entries."""
        title_norm = normalize_title_for_match(arxiv_entry["title"])
        result = resolver._stage1c_europepmc(arxiv_entry, arxiv_detection, title_norm)
        assert result is None

    def test_stage1c_returns_record_or_none(self, resolver, biorxiv_entry, biorxiv_detection):
        """Stage 1c should return PublishedRecord or None for bioRxiv."""
        title_norm = normalize_title_for_match(biorxiv_entry["title"])
        result = resolver._stage1c_europepmc(biorxiv_entry, biorxiv_detection, title_norm)
        assert result is None or isinstance(result, PublishedRecord)

    def test_stage1c_requires_title(self, resolver, biorxiv_detection):
        """Stage 1c should return None when title is empty."""
        entry = {"title": "", "author": "Jones, Alice", "journal": "bioRxiv", "doi": "10.1101/2024.01.01.123456"}
        result = resolver._stage1c_europepmc(entry, biorxiv_detection, "")
        assert result is None

    def test_stage1c_requires_author(self, resolver, biorxiv_detection):
        """Stage 1c should return None when author is missing."""
        entry = {"title": "Some Paper", "journal": "bioRxiv", "doi": "10.1101/2024.01.01.123456"}
        title_norm = normalize_title_for_match("Some Paper")
        result = resolver._stage1c_europepmc(entry, biorxiv_detection, title_norm)
        assert result is None


class TestEuropePMCMockedResponses:
    """Tests with mocked HTTP responses for Europe PMC resolver methods."""

    @pytest.fixture
    def epmc_response_json(self):
        """A valid Europe PMC search response."""
        return {
            "resultList": {
                "result": [
                    {
                        "doi": "10.1038/s41586-024-00001-1",
                        "title": "Novel Gene Discovery in Human Cells",
                        "pubYear": "2024",
                        "journalTitle": "Nature",
                        "authorString": "Jones A, Brown B",
                        "source": "MED",
                        "journalVolume": "625",
                        "journalIssueId": "7995",
                        "pageInfo": "100-110",
                    }
                ]
            }
        }

    @pytest.fixture
    def mock_resolver(self, epmc_response_json, logger):
        """Create a Resolver with mocked HTTP returning valid Europe PMC response."""
        from unittest.mock import MagicMock

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = epmc_response_json
        mock_http._request.return_value = mock_resp
        return Resolver(http=mock_http, logger=logger, scholarly_client=None)

    def test_europepmc_search_published_with_valid_response(self, mock_resolver):
        """europepmc_search_published should return record for valid response."""
        result = mock_resolver.europepmc_search_published("Novel Gene Discovery", "Jones")
        assert result is not None
        assert isinstance(result, PublishedRecord)
        assert result.doi == "10.1038/s41586-024-00001-1"
        assert result.journal == "Nature"

    def test_europepmc_search_published_empty_results(self, logger):
        """europepmc_search_published should return None for empty results."""
        from unittest.mock import MagicMock

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"resultList": {"result": []}}
        mock_http._request.return_value = mock_resp
        resolver = Resolver(http=mock_http, logger=logger, scholarly_client=None)
        result = resolver.europepmc_search_published("Nonexistent Paper", "Nobody")
        assert result is None

    def test_europepmc_search_published_404(self, logger):
        """europepmc_search_published should return None on 404."""
        from unittest.mock import MagicMock

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_http._request.return_value = mock_resp
        resolver = Resolver(http=mock_http, logger=logger, scholarly_client=None)
        result = resolver.europepmc_search_published("Some Paper", "Author")
        assert result is None

    def test_europepmc_search_published_exception(self, logger):
        """europepmc_search_published should return None on exception."""
        from unittest.mock import MagicMock

        mock_http = MagicMock()
        mock_http._request.side_effect = ConnectionError("Network error")
        resolver = Resolver(http=mock_http, logger=logger, scholarly_client=None)
        result = resolver.europepmc_search_published("Some Paper", "Author")
        assert result is None


class TestEuropePMCConverterEdgeCases:
    """Edge case tests for europepmc_result_to_record."""

    def test_ppr_source_rejected(self):
        """PPR (preprint) source should be rejected."""
        result = {
            "doi": "10.1101/2024.01.01.000001",
            "title": "Preprint Paper",
            "pubYear": "2024",
            "journalTitle": "bioRxiv",
            "authorString": "Smith J",
            "source": "PPR",
        }
        assert europepmc_result_to_record(result) is None

    def test_med_source_accepted(self):
        """MED source should be accepted."""
        result = {
            "doi": "10.1234/test.001",
            "title": "Published Paper",
            "pubYear": "2024",
            "journalTitle": "Nature",
            "authorString": "Doe J",
            "source": "MED",
            "journalVolume": "1",
        }
        rec = europepmc_result_to_record(result)
        assert rec is not None
        assert rec.journal == "Nature"

    def test_multiple_authors_parsed(self):
        """Should correctly parse multiple authors."""
        result = {
            "doi": "10.1234/test",
            "title": "Multi Author Paper",
            "pubYear": "2024",
            "journalTitle": "Science",
            "authorString": "Smith JA, Jones B, Brown CD, Williams E",
            "source": "MED",
        }
        rec = europepmc_result_to_record(result)
        assert rec is not None
        assert len(rec.authors) == 4
        assert rec.authors[0]["family"] == "Smith"
        assert rec.authors[2]["family"] == "Brown"

    def test_year_parsing(self):
        """Should parse year as integer."""
        result = {
            "doi": "10.1234/test",
            "title": "Year Paper",
            "pubYear": "2023",
            "journalTitle": "Journal",
            "authorString": "Author A",
            "source": "MED",
        }
        rec = europepmc_result_to_record(result)
        assert rec is not None
        assert rec.year == 2023
        assert isinstance(rec.year, int)


class TestIsBiorxivOrMedrxiv:
    """Tests for _is_biorxiv_or_medrxiv detection across both resolvers."""

    @pytest.fixture
    def resolver(self, fake_http, logger):
        return Resolver(http=fake_http, logger=logger, scholarly_client=None)

    def test_detects_biorxiv_doi(self, resolver):
        """Should detect 10.1101/ DOI prefix."""
        entry = {"title": "Test"}
        detection = PreprintDetection(is_preprint=True, reason="doi", doi="10.1101/2024.01.01.000001")
        assert resolver._is_biorxiv_or_medrxiv(entry, detection) is True

    def test_detects_biorxiv_note(self, resolver):
        """Should detect bioRxiv in note field."""
        entry = {"note": "bioRxiv preprint"}
        detection = PreprintDetection(is_preprint=True, reason="test", doi=None)
        assert resolver._is_biorxiv_or_medrxiv(entry, detection) is True

    def test_detects_medrxiv_publisher(self, resolver):
        """Should detect medRxiv in publisher field."""
        entry = {"publisher": "Cold Spring Harbor Laboratory (medRxiv)"}
        detection = PreprintDetection(is_preprint=True, reason="test", doi=None)
        assert resolver._is_biorxiv_or_medrxiv(entry, detection) is True

    def test_case_insensitive_detection(self, resolver):
        """Should detect case-insensitively."""
        entry = {"journal": "BIORXIV"}
        detection = PreprintDetection(is_preprint=True, reason="test", doi=None)
        assert resolver._is_biorxiv_or_medrxiv(entry, detection) is True

    def test_rejects_arxiv(self, resolver):
        """Should not detect arXiv as bioRxiv."""
        entry = {"journal": "arXiv preprint"}
        detection = PreprintDetection(is_preprint=True, reason="test", arxiv_id="2301.00001", doi=None)
        assert resolver._is_biorxiv_or_medrxiv(entry, detection) is False

    def test_rejects_regular_journal(self, resolver):
        """Should not detect regular journals."""
        entry = {"journal": "Nature"}
        detection = PreprintDetection(is_preprint=True, reason="test", doi="10.1038/test")
        assert resolver._is_biorxiv_or_medrxiv(entry, detection) is False


class TestAsyncResolverEuropePMC:
    """Tests for AsyncResolver Europe PMC methods."""

    def test_async_resolver_has_europepmc_methods(self):
        """AsyncResolver should have Europe PMC methods."""
        from bibtex_updater import AsyncResolver

        assert hasattr(AsyncResolver, "europepmc_search_published")
        assert hasattr(AsyncResolver, "_is_biorxiv_or_medrxiv")

    def test_europepmc_method_is_coroutine(self):
        """europepmc_search_published on AsyncResolver should be a coroutine."""
        import inspect

        from bibtex_updater import AsyncResolver

        assert inspect.iscoroutinefunction(AsyncResolver.europepmc_search_published)

    def test_is_biorxiv_or_medrxiv_on_async_resolver(self):
        """AsyncResolver._is_biorxiv_or_medrxiv should work the same as sync."""
        from bibtex_updater import AsyncResolver

        entry = {"journal": "bioRxiv"}
        detection = PreprintDetection(is_preprint=True, reason="test", doi="10.1101/2024.01.01.000001")
        assert AsyncResolver._is_biorxiv_or_medrxiv(entry, detection) is True

        entry_arxiv = {"journal": "arXiv preprint"}
        detection_arxiv = PreprintDetection(is_preprint=True, reason="test", arxiv_id="2301.00001", doi=None)
        assert AsyncResolver._is_biorxiv_or_medrxiv(entry_arxiv, detection_arxiv) is False
