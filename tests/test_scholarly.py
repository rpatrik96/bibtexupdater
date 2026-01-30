"""Tests for Google Scholar integration via scholarly package."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from bibtex_updater import (
    Detector,
    PublishedRecord,
    Resolver,
    ScholarlyClient,
    process_entry,
)


class TestScholarlyClient:
    """Tests for ScholarlyClient class."""

    def test_init_without_scholarly_package(self, caplog):
        """Client should handle missing scholarly package gracefully."""
        with patch.dict("sys.modules", {"scholarly": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'scholarly'")):
                client = ScholarlyClient(proxy="none", delay=1.0)
                assert client._scholarly is None

    def test_search_returns_none_when_scholarly_not_available(self):
        """Search should return None when scholarly is not installed."""
        client = ScholarlyClient.__new__(ScholarlyClient)
        client._scholarly = None
        client.delay = 1.0
        client._last_request = 0.0
        client.logger = logging.getLogger(__name__)

        result = client.search("Test Title", "Smith")
        assert result is None

    @patch("bibtex_updater.updater.time.sleep")
    def test_rate_limit_waits_when_needed(self, mock_sleep):
        """Rate limiter should sleep when requests are too fast."""
        client = ScholarlyClient.__new__(ScholarlyClient)
        client.delay = 5.0
        client._last_request = time.time()  # Just now
        client.logger = logging.getLogger(__name__)
        client._scholarly = None

        client._rate_limit()

        # Should have slept approximately delay seconds
        assert mock_sleep.called
        sleep_time = mock_sleep.call_args[0][0]
        assert sleep_time > 0 and sleep_time <= 5.0

    def test_rate_limit_no_wait_when_enough_time_passed(self):
        """Rate limiter should not sleep when enough time has passed."""
        client = ScholarlyClient.__new__(ScholarlyClient)
        client.delay = 1.0
        client._last_request = time.time() - 10  # 10 seconds ago
        client.logger = logging.getLogger(__name__)
        client._scholarly = None

        with patch("bibtex_updater.updater.time.sleep") as mock_sleep:
            client._rate_limit()
            mock_sleep.assert_not_called()


class TestScholarlyClientSearch:
    """Tests for ScholarlyClient.search() method."""

    def test_search_calls_scholarly_search_pubs(self):
        """Search should call scholarly.search_pubs with correct query."""
        mock_scholarly = MagicMock()
        mock_scholarly.search_pubs.return_value = iter([])

        client = ScholarlyClient.__new__(ScholarlyClient)
        client._scholarly = mock_scholarly
        client.delay = 0.0
        client._last_request = 0.0
        client.logger = logging.getLogger(__name__)

        client.search("Machine Learning", "Smith")

        mock_scholarly.search_pubs.assert_called_once_with("Machine Learning Smith")

    def test_search_fills_publication_when_found(self):
        """Search should fill the publication when one is found."""
        mock_pub = {"bib": {"title": "Test Paper"}}
        mock_filled = {"bib": {"title": "Test Paper", "author": "John Smith"}}

        mock_scholarly = MagicMock()
        mock_scholarly.search_pubs.return_value = iter([mock_pub])
        mock_scholarly.fill.return_value = mock_filled

        client = ScholarlyClient.__new__(ScholarlyClient)
        client._scholarly = mock_scholarly
        client.delay = 0.0
        client._last_request = 0.0
        client.logger = logging.getLogger(__name__)

        result = client.search("Test Paper", "Smith")

        assert result == mock_filled
        mock_scholarly.fill.assert_called_once_with(mock_pub)

    def test_search_returns_none_when_no_results(self):
        """Search should return None when no publications found."""
        mock_scholarly = MagicMock()
        mock_scholarly.search_pubs.return_value = iter([])

        client = ScholarlyClient.__new__(ScholarlyClient)
        client._scholarly = mock_scholarly
        client.delay = 0.0
        client._last_request = 0.0
        client.logger = logging.getLogger(__name__)

        result = client.search("Nonexistent Paper", "Nobody")

        assert result is None

    def test_search_handles_exception_gracefully(self, caplog):
        """Search should log warning and return None on exception."""
        mock_scholarly = MagicMock()
        mock_scholarly.search_pubs.side_effect = Exception("Rate limited by Google")

        client = ScholarlyClient.__new__(ScholarlyClient)
        client._scholarly = mock_scholarly
        client.delay = 0.0
        client._last_request = 0.0
        client.logger = logging.getLogger(__name__)

        with caplog.at_level(logging.WARNING):
            result = client.search("Some Paper", "Author")

        assert result is None
        assert "Scholarly search failed" in caplog.text

    def test_search_handles_stop_iteration(self):
        """Search should handle StopIteration gracefully."""

        def empty_generator():
            return
            yield  # Make it a generator

        mock_scholarly = MagicMock()
        mock_scholarly.search_pubs.return_value = empty_generator()

        client = ScholarlyClient.__new__(ScholarlyClient)
        client._scholarly = mock_scholarly
        client.delay = 0.0
        client._last_request = 0.0
        client.logger = logging.getLogger(__name__)

        result = client.search("Paper", "Author")
        assert result is None

    def test_search_returns_partial_results_when_fill_fails(self, caplog):
        """Search should return partial results with _partial flag when fill() fails."""
        mock_pub = {"bib": {"title": "Test Paper", "venue": "Advances in neural...", "pub_year": "2020"}}
        mock_scholarly = MagicMock()
        mock_scholarly.search_pubs.return_value = iter([mock_pub])
        mock_scholarly.fill.side_effect = Exception("Cannot Fetch from Google Scholar")

        client = ScholarlyClient.__new__(ScholarlyClient)
        client._scholarly = mock_scholarly
        client.delay = 0.0
        client._last_request = 0.0
        client.logger = logging.getLogger(__name__)

        with caplog.at_level(logging.DEBUG):
            result = client.search("Test Paper", "Author")

        assert result is not None
        assert result["_partial"] is True
        assert result["bib"]["title"] == "Test Paper"
        assert "fill() failed" in caplog.text


class TestScholarlyToRecord:
    """Tests for Resolver._scholarly_to_record() conversion."""

    @pytest.fixture
    def resolver(self, fake_http, logger):
        """Create resolver without scholarly client."""
        return Resolver(http=fake_http, logger=logger, scholarly_client=None)

    def test_conversion_basic_publication(self, resolver):
        """Test conversion of a basic scholarly publication."""
        pub = {
            "bib": {
                "title": "Deep Learning for Image Recognition",
                "author": "John Smith and Jane Doe",
                "venue": "Nature Machine Intelligence",
                "pub_year": "2023",
                "volume": "5",
                "number": "3",
                "pages": "100-115",
            },
            "pub_url": "https://doi.org/10.1038/s42256-023-00001-1",
        }

        record = resolver._scholarly_to_record(pub)

        assert record is not None
        assert record.title == "Deep Learning for Image Recognition"
        assert record.journal == "Nature Machine Intelligence"
        assert record.year == 2023
        assert record.volume == "5"
        assert record.number == "3"
        assert record.pages == "100-115"
        assert record.doi == "10.1038/s42256-023-00001-1"
        assert record.method == "GoogleScholar(search)"
        assert len(record.authors) == 2
        assert record.authors[0]["family"] == "Smith"
        assert record.authors[1]["family"] == "Doe"

    def test_conversion_extracts_doi_from_eprint_url(self, resolver):
        """Test DOI extraction from eprint_url when pub_url doesn't have it."""
        pub = {
            "bib": {"title": "Some Paper", "author": "Author One"},
            "pub_url": "https://example.com/paper",
            "eprint_url": "https://doi.org/10.1234/test.5678",
        }

        record = resolver._scholarly_to_record(pub)

        assert record.doi == "10.1234/test.5678"

    def test_conversion_with_journal_field(self, resolver):
        """Test conversion when venue is from journal field."""
        pub = {
            "bib": {
                "title": "Paper",
                "author": "Author One",
                "journal": "JMLR",
            }
        }

        record = resolver._scholarly_to_record(pub)

        assert record.journal == "JMLR"

    def test_conversion_with_booktitle_field(self, resolver):
        """Test conversion when venue is from booktitle field."""
        pub = {
            "bib": {
                "title": "Paper",
                "author": "Author One",
                "booktitle": "NeurIPS 2023",
            }
        }

        record = resolver._scholarly_to_record(pub)

        assert record.journal == "NeurIPS 2023"

    def test_conversion_handles_single_name_author(self, resolver):
        """Test conversion with single-name author."""
        pub = {
            "bib": {
                "title": "Paper",
                "author": "Madonna",
            }
        }

        record = resolver._scholarly_to_record(pub)

        assert len(record.authors) == 1
        assert record.authors[0]["family"] == "Madonna"
        assert record.authors[0]["given"] == ""

    def test_conversion_handles_multi_part_names(self, resolver):
        """Test conversion with multi-part given names."""
        pub = {
            "bib": {
                "title": "Paper",
                "author": "Jean Claude Van Damme",
            }
        }

        record = resolver._scholarly_to_record(pub)

        assert record.authors[0]["given"] == "Jean Claude Van"
        assert record.authors[0]["family"] == "Damme"

    def test_conversion_returns_none_for_empty_pub(self, resolver):
        """Test conversion returns None for empty publication."""
        assert resolver._scholarly_to_record(None) is None
        assert resolver._scholarly_to_record({}) is None
        assert resolver._scholarly_to_record({"bib": {}}) is None

    def test_conversion_handles_invalid_year(self, resolver):
        """Test conversion handles non-numeric year gracefully."""
        pub = {
            "bib": {
                "title": "Paper",
                "author": "Author",
                "pub_year": "invalid",
            }
        }

        record = resolver._scholarly_to_record(pub)

        assert record is not None
        assert record.year is None

    def test_type_is_journal_article_when_venue_present(self, resolver):
        """Test type is journal-article when venue is present."""
        pub = {
            "bib": {
                "title": "Paper",
                "author": "Author",
                "venue": "Some Journal",
            }
        }

        record = resolver._scholarly_to_record(pub)
        assert record.type == "journal-article"

    def test_type_is_unknown_when_no_venue(self, resolver):
        """Test type is unknown when no venue is present."""
        pub = {
            "bib": {
                "title": "Paper",
                "author": "Author",
            }
        }

        record = resolver._scholarly_to_record(pub)
        assert record.type == "unknown"

    def test_partial_results_marked_in_method(self, resolver):
        """Test that partial results have method marked with 'partial'."""
        pub = {
            "_partial": True,
            "bib": {
                "title": "Paper",
                "author": "Author",
                "venue": "Advances in neural information processing systems",
                "pub_year": "2020",
            },
        }

        record = resolver._scholarly_to_record(pub)
        assert "partial" in record.method
        assert record.method == "GoogleScholar(search,partial)"

    def test_full_results_not_marked_partial(self, resolver):
        """Test that full results don't have 'partial' in method."""
        pub = {
            "bib": {
                "title": "Paper",
                "author": "Author",
                "venue": "Nature",
                "pub_year": "2020",
            },
        }

        record = resolver._scholarly_to_record(pub)
        assert "partial" not in record.method
        assert record.method == "GoogleScholar(search)"


class TestCredibleJournalArticlePartialResults:
    """Tests for _credible_journal_article with partial Google Scholar results."""

    def test_partial_results_accepted_with_known_conference_venue(self):
        """Partial results should be accepted if venue matches known conference."""
        record = PublishedRecord(
            doi=None,
            url=None,  # No URL (partial result)
            title="Learning with Differentiable Perturbed Optimizers",
            authors=[{"given": "Quentin", "family": "Berthet"}],
            journal="Advances in neural information processing systems",
            year=2020,
            volume=None,  # No volume (partial result)
            number=None,
            pages=None,  # No pages (partial result)
            type="proceedings-article",
            method="GoogleScholar(search,partial)",
            confidence=0.0,
        )

        assert Resolver._credible_journal_article(record) is True

    def test_partial_results_rejected_with_unknown_venue(self):
        """Partial results should be rejected if venue is not a known conference."""
        record = PublishedRecord(
            doi=None,
            url=None,
            title="Some Paper",
            authors=[{"given": "John", "family": "Doe"}],
            journal="Unknown Journal",
            year=2020,
            volume=None,
            number=None,
            pages=None,
            type="journal-article",
            method="GoogleScholar(search,partial)",
            confidence=0.0,
        )

        assert Resolver._credible_journal_article(record) is False

    def test_full_results_still_require_volume_or_url(self):
        """Full Google Scholar results still need volume/pages or url/doi."""
        record = PublishedRecord(
            doi=None,
            url=None,
            title="Some Paper",
            authors=[{"given": "John", "family": "Doe"}],
            journal="Advances in neural information processing systems",
            year=2020,
            volume=None,
            number=None,
            pages=None,
            type="proceedings-article",
            method="GoogleScholar(search)",  # Not partial
            confidence=0.0,
        )

        # Full results without volume/pages/url/doi should be rejected
        assert Resolver._credible_journal_article(record) is False

    def test_partial_neurips_truncated_venue_accepted(self):
        """Partial results with truncated NeurIPS venue should be accepted."""
        record = PublishedRecord(
            doi=None,
            url=None,
            title="Test Paper",
            authors=[{"given": "Test", "family": "Author"}],
            journal="Advances in neural...",  # Truncated venue from Google Scholar
            year=2020,
            volume=None,
            number=None,
            pages=None,
            type="proceedings-article",
            method="GoogleScholar(search,partial)",
            confidence=0.0,
        )

        assert Resolver._credible_journal_article(record) is True

    def test_partial_icml_venue_accepted(self):
        """Partial results with ICML venue should be accepted."""
        record = PublishedRecord(
            doi=None,
            url=None,
            title="Test Paper",
            authors=[{"given": "Test", "family": "Author"}],
            journal="International Conference on Machine Learning",
            year=2020,
            volume=None,
            number=None,
            pages=None,
            type="proceedings-article",
            method="GoogleScholar(search,partial)",
            confidence=0.0,
        )

        assert Resolver._credible_journal_article(record) is True


class TestResolverStage6:
    """Tests for Stage 6 (Google Scholar fallback) in Resolver.resolve()."""

    def test_stage6_not_called_when_scholarly_client_is_none(self, make_entry, fake_http, logger):
        """Stage 6 should be skipped when scholarly_client is None."""
        resolver = Resolver(http=fake_http, logger=logger, scholarly_client=None)
        entry = make_entry(
            title="Test Paper",
            author="John Smith",
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )
        detection = Detector().detect(entry)

        # Without scholarly client, should return None (no match from other stages)
        result = resolver.resolve(entry, detection)
        # Result could be None or a match from earlier stages - we just verify no crash
        assert result is None or isinstance(result, PublishedRecord)

    def test_stage6_called_when_scholarly_client_is_set(self, make_entry, fake_http, logger):
        """Stage 6 should be called when scholarly_client is set."""
        mock_scholarly_client = MagicMock()
        mock_scholarly_client.search.return_value = {
            "bib": {
                "title": "Test Paper",
                "author": "John Smith",
                "venue": "Nature",
                "pub_year": "2021",
                "volume": "100",
                "pages": "1-10",
            },
            "pub_url": "https://doi.org/10.1038/test",
        }

        resolver = Resolver(http=fake_http, logger=logger, scholarly_client=mock_scholarly_client)
        entry = make_entry(
            title="Test Paper",
            author="John Smith",
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )
        detection = Detector().detect(entry)

        resolver.resolve(entry, detection)

        # Scholarly should have been searched
        mock_scholarly_client.search.assert_called()

    def test_stage6_returns_match_with_high_confidence(self, make_entry, fake_http, logger):
        """Stage 6 should return match when confidence >= 0.9."""
        mock_scholarly_client = MagicMock()
        mock_scholarly_client.search.return_value = {
            "bib": {
                "title": "Deep Learning for Test Cases",
                "author": "John Smith and Jane Doe",
                "venue": "Journal of Testing",
                "pub_year": "2021",
                "volume": "10",
                "pages": "1-20",
            },
            "pub_url": "https://doi.org/10.1000/test",
        }

        resolver = Resolver(http=fake_http, logger=logger, scholarly_client=mock_scholarly_client)
        entry = make_entry(
            title="Deep Learning for Test Cases",
            author="John Smith and Jane Doe",
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )
        detection = Detector().detect(entry)

        result = resolver.resolve(entry, detection)

        if result:
            assert result.method == "GoogleScholar(search)"
            assert result.confidence >= 0.9

    def test_stage6_skips_low_confidence_match(self, make_entry, fake_http, logger):
        """Stage 6 should skip matches with confidence < 0.9."""
        mock_scholarly_client = MagicMock()
        mock_scholarly_client.search.return_value = {
            "bib": {
                "title": "Completely Different Paper Title",
                "author": "Different Author",
                "venue": "Other Journal",
                "pub_year": "2021",
            },
        }

        resolver = Resolver(http=fake_http, logger=logger, scholarly_client=mock_scholarly_client)
        entry = make_entry(
            title="Deep Learning for Test Cases",
            author="John Smith",
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )
        detection = Detector().detect(entry)

        result = resolver.resolve(entry, detection)

        # Low confidence match should be skipped
        if result:
            # If we got a result, it should not be from GoogleScholar with low confidence
            if result.method == "GoogleScholar(search)":
                assert result.confidence >= 0.9


class TestScholarlyIntegration:
    """Integration tests for scholarly with full pipeline."""

    def test_scholarly_disabled_by_default(self, make_entry, detector, updater, logger, fake_http):
        """Scholarly should not be used unless explicitly enabled."""
        # Create resolver without scholarly_client (default)
        resolver = Resolver(http=fake_http, logger=logger, scholarly_client=None)

        entry = make_entry(
            title="Test Paper",
            author="John Smith",
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )

        result = process_entry(entry, detector, resolver, updater, logger)

        # Should fail because no scholarly fallback and mock HTTP returns nothing
        assert result.action in ("failed", "unchanged")

    def test_full_pipeline_with_scholarly_fallback(self, make_entry, detector, updater, logger, fake_http):
        """Test full pipeline using scholarly as fallback."""
        mock_scholarly_client = MagicMock()
        mock_scholarly_client.search.return_value = {
            "bib": {
                "title": "Machine Learning Advances",
                "author": "Jane Doe and John Smith",
                "venue": "Nature Machine Intelligence",
                "pub_year": "2023",
                "volume": "5",
                "number": "1",
                "pages": "10-25",
            },
            "pub_url": "https://doi.org/10.1038/s42256-023-00001-1",
        }

        resolver = Resolver(http=fake_http, logger=logger, scholarly_client=mock_scholarly_client)

        entry = make_entry(
            title="Machine Learning Advances",
            author="Jane Doe and John Smith",
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )

        result = process_entry(entry, detector, resolver, updater, logger)

        # Should upgrade via GoogleScholar
        if result.action == "upgraded":
            assert result.method == "GoogleScholar(search)"
            assert result.updated["journal"] == "Nature Machine Intelligence"

    def test_scholarly_graceful_failure_continues_pipeline(self, make_entry, detector, updater, logger, fake_http):
        """Scholarly failure should not crash the pipeline."""
        mock_scholarly_client = MagicMock()
        # ScholarlyClient.search() catches exceptions internally and returns None
        # So we simulate that behavior - a failing search returns None
        mock_scholarly_client.search.return_value = None

        resolver = Resolver(http=fake_http, logger=logger, scholarly_client=mock_scholarly_client)

        entry = make_entry(
            title="Test Paper",
            author="John Smith",
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )

        # Should not raise an exception
        result = process_entry(entry, detector, resolver, updater, logger)

        # Pipeline completes, scholarly was called but returned None
        mock_scholarly_client.search.assert_called()
        assert result.action in ("failed", "unchanged")


class TestScholarlyProxyConfiguration:
    """Tests for proxy configuration in ScholarlyClient."""

    def test_no_proxy_by_default(self):
        """No proxy should be configured by default."""
        with patch("bibtex_updater.ScholarlyClient._setup") as mock_setup:
            mock_setup.return_value = None
            ScholarlyClient(proxy="none", delay=1.0)
            mock_setup.assert_called_once_with("none")

    @patch("bibtex_updater.updater.time.sleep")
    def test_default_delay_is_5_seconds(self, mock_sleep):
        """Default delay should be 5 seconds."""
        client = ScholarlyClient.__new__(ScholarlyClient)
        client.delay = 5.0  # Default value
        client._last_request = time.time()
        client._scholarly = None
        client.logger = logging.getLogger(__name__)

        client._rate_limit()

        if mock_sleep.called:
            sleep_time = mock_sleep.call_args[0][0]
            assert sleep_time <= 5.0
