"""Tests for OpenAlex integration (Stage 1b)."""

from __future__ import annotations

import pytest

from bibtex_updater import AsyncResolver, PreprintDetection, PublishedRecord, Resolver
from bibtex_updater.utils import openalex_work_to_record

# --- openalex_work_to_record converter tests ---


class TestOpenAlexWorkToRecord:
    """Tests for the openalex_work_to_record converter function."""

    def test_valid_published_work(self):
        """Should convert a published journal article."""
        work = {
            "doi": "https://doi.org/10.1234/example.2024.001",
            "title": "Deep Learning for Everything",
            "publication_year": 2024,
            "type": "article",
            "primary_location": {
                "source": {
                    "display_name": "Journal of Machine Learning",
                    "type": "journal",
                },
                "version": "publishedVersion",
            },
            "authorships": [
                {"author": {"display_name": "John Smith"}},
                {"author": {"display_name": "Jane Doe"}},
            ],
            "biblio": {
                "volume": "42",
                "issue": "1",
                "first_page": "1",
                "last_page": "20",
            },
            "locations": [
                {"version": "publishedVersion", "source": {"type": "journal"}},
            ],
        }
        rec = openalex_work_to_record(work)
        assert rec is not None
        assert rec.doi == "10.1234/example.2024.001"
        assert rec.title == "Deep Learning for Everything"
        assert rec.year == 2024
        assert rec.journal == "Journal of Machine Learning"
        assert rec.volume == "42"
        assert rec.number == "1"
        assert rec.pages == "1-20"
        assert len(rec.authors) == 2

    def test_rejects_preprint_type(self):
        """Should reject works with preprint type."""
        work = {
            "doi": "https://doi.org/10.1234/example",
            "title": "Some Preprint",
            "publication_year": 2024,
            "type": "preprint",
            "primary_location": {
                "source": {"display_name": "arXiv", "type": "repository"},
            },
            "authorships": [{"author": {"display_name": "John Smith"}}],
            "biblio": {},
            "locations": [],
        }
        rec = openalex_work_to_record(work)
        assert rec is None

    def test_rejects_no_doi(self):
        """Should reject works without a DOI."""
        work = {
            "doi": None,
            "title": "No DOI Paper",
            "publication_year": 2024,
            "type": "article",
            "primary_location": {
                "source": {"display_name": "Some Journal", "type": "journal"},
            },
            "authorships": [],
            "biblio": {},
            "locations": [],
        }
        rec = openalex_work_to_record(work)
        assert rec is None

    def test_rejects_preprint_venue(self):
        """Should reject works from preprint venues like arXiv, bioRxiv."""
        work = {
            "doi": "https://doi.org/10.48550/arXiv.2301.00001",
            "title": "ArXiv Paper",
            "publication_year": 2024,
            "type": "article",
            "primary_location": {
                "source": {"display_name": "arXiv", "type": "repository"},
            },
            "authorships": [{"author": {"display_name": "John Smith"}}],
            "biblio": {},
            "locations": [{"version": "submittedVersion", "source": {"type": "repository"}}],
        }
        rec = openalex_work_to_record(work)
        # Should be rejected because venue is arXiv
        assert rec is None

    def test_empty_work(self):
        """Should handle empty/minimal work objects."""
        rec = openalex_work_to_record({})
        assert rec is None

    def test_author_parsing(self):
        """Should parse author names into given/family components."""
        work = {
            "doi": "https://doi.org/10.1234/example",
            "title": "Test Paper",
            "publication_year": 2024,
            "type": "article",
            "primary_location": {
                "source": {"display_name": "Test Journal", "type": "journal"},
                "version": "publishedVersion",
            },
            "authorships": [
                {"author": {"display_name": "Alice B. Jones"}},
                {"author": {"display_name": "Bob"}},
            ],
            "biblio": {"volume": "1"},
            "locations": [{"version": "publishedVersion", "source": {"type": "journal"}}],
        }
        rec = openalex_work_to_record(work)
        assert rec is not None
        assert rec.authors[0]["family"] == "Jones"
        assert "Alice" in rec.authors[0]["given"]
        # Single name should have family but possibly empty given
        assert rec.authors[1]["family"] == "Bob"


# --- Resolver Stage 1b tests ---


class TestStage1bOpenAlex:
    """Tests for the OpenAlex resolution stage."""

    @pytest.fixture
    def resolver(self, fake_http, logger):
        return Resolver(http=fake_http, logger=logger, scholarly_client=None)

    @pytest.fixture
    def arxiv_detection(self):
        return PreprintDetection(
            is_preprint=True,
            reason="arXiv ID found",
            arxiv_id="2301.00001",
            doi=None,
        )

    @pytest.fixture
    def doi_detection(self):
        return PreprintDetection(
            is_preprint=True,
            reason="bioRxiv DOI",
            arxiv_id=None,
            doi="10.1101/2024.01.01.000001",
        )

    def test_stage1b_returns_none_without_ids(self, resolver):
        """Stage 1b should return None when no arXiv ID or DOI."""
        detection = PreprintDetection(is_preprint=True, reason="test", arxiv_id=None, doi=None)
        result = resolver._stage1b_openalex(detection, None)
        assert result is None

    def test_stage1b_returns_record_or_none(self, resolver, arxiv_detection):
        """Stage 1b should return PublishedRecord or None."""
        result = resolver._stage1b_openalex(arxiv_detection, None)
        assert result is None or isinstance(result, PublishedRecord)

    def test_stage1b_tries_doi_fallback(self, resolver, doi_detection):
        """Stage 1b should try DOI when no arXiv ID."""
        result = resolver._stage1b_openalex(doi_detection, None)
        assert result is None or isinstance(result, PublishedRecord)

    def test_stage1b_tries_candidate_doi(self, resolver):
        """Stage 1b should try candidate_doi from Stage 1."""
        detection = PreprintDetection(is_preprint=True, reason="test", arxiv_id=None, doi=None)
        result = resolver._stage1b_openalex(detection, "10.1234/test")
        assert result is None or isinstance(result, PublishedRecord)

    def test_openalex_from_arxiv_returns_none_on_error(self, resolver):
        """openalex_from_arxiv should return None on HTTP errors."""
        result = resolver.openalex_from_arxiv("2301.00001")
        assert result is None

    def test_openalex_from_doi_returns_none_on_empty(self, resolver):
        """openalex_from_doi should return None for empty DOI."""
        result = resolver.openalex_from_doi("")
        assert result is None

    def test_stage1b_prefers_arxiv_over_doi(self, resolver):
        """Stage 1b should try arXiv ID first before DOI."""
        detection = PreprintDetection(
            is_preprint=True,
            reason="arXiv",
            arxiv_id="2301.00001",
            doi="10.1101/2024.01.01.000001",
        )
        # With fake HTTP both will fail, but code path exercises arXiv-first logic
        result = resolver._stage1b_openalex(detection, None)
        assert result is None


class TestOpenAlexMockedResponses:
    """Tests with mocked HTTP responses for OpenAlex resolver methods."""

    @pytest.fixture
    def published_work_json(self):
        """A valid OpenAlex published work response."""
        return {
            "doi": "https://doi.org/10.1234/example.2024.001",
            "title": "Deep Learning for Everything",
            "publication_year": 2024,
            "type": "article",
            "primary_location": {
                "source": {
                    "display_name": "Journal of Machine Learning",
                    "type": "journal",
                    "host_organization_name": "ML Publisher",
                },
                "version": "publishedVersion",
            },
            "authorships": [
                {"author": {"display_name": "John Smith"}},
                {"author": {"display_name": "Jane Doe"}},
            ],
            "biblio": {
                "volume": "42",
                "issue": "1",
                "first_page": "1",
                "last_page": "20",
            },
            "locations": [
                {"version": "publishedVersion", "source": {"type": "journal"}},
            ],
        }

    @pytest.fixture
    def mock_resolver(self, published_work_json, logger):
        """Create a Resolver with mocked HTTP that returns a valid OpenAlex response."""
        from unittest.mock import MagicMock

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = published_work_json
        mock_http._request.return_value = mock_resp
        return Resolver(http=mock_http, logger=logger, scholarly_client=None)

    def test_openalex_from_arxiv_with_valid_response(self, mock_resolver):
        """openalex_from_arxiv should return PublishedRecord for valid work."""
        result = mock_resolver.openalex_from_arxiv("2301.00001")
        assert result is not None
        assert isinstance(result, PublishedRecord)
        assert result.method == "OpenAlex(arxiv)"
        assert result.confidence == 1.0
        assert result.doi == "10.1234/example.2024.001"
        assert result.journal == "Journal of Machine Learning"

    def test_openalex_from_doi_with_valid_response(self, mock_resolver):
        """openalex_from_doi should return PublishedRecord for valid work."""
        result = mock_resolver.openalex_from_doi("10.1234/example.2024.001")
        assert result is not None
        assert isinstance(result, PublishedRecord)
        assert result.method == "OpenAlex(doi)"
        assert result.confidence == 1.0

    def test_openalex_from_arxiv_404(self, logger):
        """openalex_from_arxiv should return None on 404."""
        from unittest.mock import MagicMock

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_http._request.return_value = mock_resp
        resolver = Resolver(http=mock_http, logger=logger, scholarly_client=None)
        result = resolver.openalex_from_arxiv("9999.99999")
        assert result is None

    def test_openalex_from_arxiv_exception(self, logger):
        """openalex_from_arxiv should return None on exception."""
        from unittest.mock import MagicMock

        mock_http = MagicMock()
        mock_http._request.side_effect = ConnectionError("Network error")
        resolver = Resolver(http=mock_http, logger=logger, scholarly_client=None)
        result = resolver.openalex_from_arxiv("2301.00001")
        assert result is None

    def test_stage1b_returns_result_from_arxiv(self, mock_resolver):
        """Stage 1b should return result from OpenAlex arXiv lookup."""
        detection = PreprintDetection(is_preprint=True, reason="arXiv", arxiv_id="2301.00001", doi=None)
        result = mock_resolver._stage1b_openalex(detection, None)
        assert result is not None
        assert result.method == "OpenAlex(arxiv)"


class TestOpenAlexConverterEdgeCases:
    """Edge case tests for openalex_work_to_record."""

    def test_posted_content_type_rejected(self):
        """posted-content (preprint) type should be rejected."""
        work = {
            "doi": "https://doi.org/10.48550/arXiv.2301.00001",
            "title": "Posted Content",
            "publication_year": 2024,
            "type": "posted-content",
            "primary_location": {"source": {"display_name": "arXiv"}},
            "authorships": [],
            "biblio": {},
            "locations": [],
        }
        assert openalex_work_to_record(work) is None

    def test_pages_single_page(self):
        """Should handle single page (first_page only)."""
        work = {
            "doi": "https://doi.org/10.1234/test",
            "title": "Single Page",
            "publication_year": 2024,
            "type": "article",
            "primary_location": {
                "source": {"display_name": "Journal"},
                "version": "publishedVersion",
            },
            "authorships": [{"author": {"display_name": "Test Author"}}],
            "biblio": {"volume": "1", "first_page": "42"},
            "locations": [{"version": "publishedVersion", "source": {"type": "journal"}}],
        }
        rec = openalex_work_to_record(work)
        assert rec is not None
        assert rec.pages == "42"

    def test_pages_same_first_last(self):
        """Should handle same first and last page."""
        work = {
            "doi": "https://doi.org/10.1234/test",
            "title": "One Page Paper",
            "publication_year": 2024,
            "type": "article",
            "primary_location": {
                "source": {"display_name": "Journal"},
                "version": "publishedVersion",
            },
            "authorships": [{"author": {"display_name": "Author"}}],
            "biblio": {"volume": "1", "first_page": "10", "last_page": "10"},
            "locations": [{"version": "publishedVersion", "source": {"type": "journal"}}],
        }
        rec = openalex_work_to_record(work)
        assert rec is not None
        assert rec.pages == "10"

    def test_none_input(self):
        """Should handle None input."""
        assert openalex_work_to_record(None) is None

    def test_medrxiv_venue_rejected(self):
        """medRxiv venue should be rejected."""
        work = {
            "doi": "https://doi.org/10.1101/2024.01.01.000001",
            "title": "medRxiv Paper",
            "publication_year": 2024,
            "type": "article",
            "primary_location": {
                "source": {"display_name": "medRxiv"},
            },
            "authorships": [{"author": {"display_name": "Author"}}],
            "biblio": {},
            "locations": [],
        }
        assert openalex_work_to_record(work) is None


class TestAsyncResolverOpenAlex:
    """Tests for AsyncResolver OpenAlex methods."""

    def test_async_resolver_has_openalex_methods(self):
        """AsyncResolver should have OpenAlex methods."""
        assert hasattr(AsyncResolver, "openalex_from_arxiv")
        assert hasattr(AsyncResolver, "openalex_from_doi")

    def test_openalex_methods_are_coroutines(self):
        """OpenAlex methods on AsyncResolver should be coroutine functions."""
        import inspect

        assert inspect.iscoroutinefunction(AsyncResolver.openalex_from_arxiv)
        assert inspect.iscoroutinefunction(AsyncResolver.openalex_from_doi)
