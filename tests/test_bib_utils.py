"""Tests for bib_utils shared module.

These tests focus on API record conversion functions that are new to bib_utils.
Core utility functions are tested in test_utils.py.
"""

from __future__ import annotations

from bibtex_updater.utils import (
    DiskCache,
    RateLimiter,
    crossref_message_to_record,
    dblp_hit_to_record,
    s2_data_to_record,
)


class TestCrossrefMessageToRecord:
    """Tests for crossref_message_to_record function."""

    def test_basic_conversion(self):
        msg = {
            "DOI": "10.1234/test",
            "type": "journal-article",
            "title": ["Test Title"],
            "author": [{"given": "John", "family": "Smith"}],
            "container-title": ["Test Journal"],
            "published-print": {"date-parts": [[2021, 3, 15]]},
            "volume": "42",
            "issue": "3",
            "page": "100-120",
            "publisher": "Test Publisher",
            "URL": "https://doi.org/10.1234/test",
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.doi == "10.1234/test"
        assert rec.title == "Test Title"
        assert rec.journal == "Test Journal"
        assert rec.year == 2021
        assert rec.volume == "42"
        assert rec.number == "3"
        assert rec.pages == "100-120"
        assert len(rec.authors) == 1
        assert rec.authors[0]["family"] == "Smith"

    def test_no_doi_returns_none(self):
        msg = {"title": ["Test"], "author": []}
        rec = crossref_message_to_record(msg)
        assert rec is None

    def test_html_stripped_from_title(self):
        msg = {
            "DOI": "10.1234/test",
            "title": ["<i>Italic</i> and <b>Bold</b> text"],
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert "<i>" not in rec.title
        assert "<b>" not in rec.title

    def test_literal_author_handling(self):
        msg = {
            "DOI": "10.1234/test",
            "author": [{"literal": "John Smith"}, {"given": "Jane", "family": "Doe"}],
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert len(rec.authors) == 2
        assert rec.authors[0]["family"] == "Smith"
        assert rec.authors[0]["given"] == "John"
        assert rec.authors[1]["family"] == "Doe"

    def test_multiple_date_fields(self):
        # published-online should be used if published-print is missing
        msg = {
            "DOI": "10.1234/test",
            "published-online": {"date-parts": [[2020, 6, 1]]},
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.year == 2020

    def test_issued_date_fallback(self):
        msg = {
            "DOI": "10.1234/test",
            "issued": {"date-parts": [[2019, 1, 1]]},
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.year == 2019

    def test_journal_issue_nested(self):
        msg = {
            "DOI": "10.1234/test",
            "journal-issue": {"issue": "5"},
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.number == "5"

    def test_empty_author_list(self):
        msg = {
            "DOI": "10.1234/test",
            "author": [],
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.authors == []


class TestDblpHitToRecord:
    """Tests for dblp_hit_to_record function."""

    def test_basic_conversion(self):
        hit = {
            "info": {
                "title": "Test Paper Title",
                "authors": {"author": ["John Smith", "Jane Doe"]},
                "venue": "Journal of Testing",
                "year": "2021",
                "doi": "10.1234/dblp",
                "volume": "10",
                "pages": "1-15",
                "type": "Journal Articles",
            }
        }
        rec = dblp_hit_to_record(hit)
        assert rec is not None
        assert rec.title == "Test Paper Title"
        assert rec.journal == "Journal of Testing"
        assert rec.year == 2021
        assert len(rec.authors) == 2

    def test_html_stripped_from_title(self):
        hit = {
            "info": {
                "title": "<i>Emphasized</i> Title",
                "authors": {"author": ["Author"]},
                "venue": "Journal",
                "year": "2020",
                "doi": "10.1234/test",
                "type": "Journal Articles",
            }
        }
        rec = dblp_hit_to_record(hit)
        assert rec is not None
        assert "<i>" not in rec.title

    def test_single_author_dict_format(self):
        hit = {
            "info": {
                "title": "Test",
                "authors": {"author": {"text": "Single Author"}},
                "venue": "Journal",
                "year": "2021",
                "ee": "https://example.com",
                "type": "Journal Articles",
            }
        }
        rec = dblp_hit_to_record(hit)
        assert rec is not None
        assert len(rec.authors) == 1

    def test_conference_accepted(self):
        """Conference papers should be accepted with proceedings-article type."""
        hit = {
            "info": {
                "title": "Conference Paper",
                "authors": {"author": ["Author"]},
                "venue": "Proceedings of Conference",
                "year": "2021",
                "doi": "10.1234/conf",
                "type": "Conference and Workshop Papers",
            }
        }
        rec = dblp_hit_to_record(hit)
        assert rec is not None
        assert rec.type == "proceedings-article"
        assert rec.title == "Conference Paper"
        assert rec.journal == "Proceedings of Conference"
        assert rec.year == 2021

    def test_no_venue_returns_none(self):
        hit = {
            "info": {
                "title": "Test",
                "authors": {"author": ["Author"]},
                "year": "2021",
                "doi": "10.1234/test",
                "type": "Journal Articles",
            }
        }
        rec = dblp_hit_to_record(hit)
        assert rec is None

    def test_no_doi_or_url_returns_none(self):
        hit = {
            "info": {
                "title": "Test",
                "authors": {"author": ["Author"]},
                "venue": "Journal",
                "year": "2021",
                "type": "Journal Articles",
            }
        }
        rec = dblp_hit_to_record(hit)
        assert rec is None

    def test_url_fallback_when_no_doi(self):
        hit = {
            "info": {
                "title": "Test",
                "authors": {"author": ["Author"]},
                "venue": "Journal",
                "year": "2021",
                "ee": "https://example.com/paper",
                "type": "Journal Articles",
            }
        }
        rec = dblp_hit_to_record(hit)
        assert rec is not None
        assert rec.url == "https://example.com/paper"


class TestS2DataToRecord:
    """Tests for s2_data_to_record function."""

    def test_basic_conversion(self):
        data = {
            "doi": "10.1234/s2test",
            "title": "Semantic Scholar Paper",
            "authors": [{"name": "John Smith"}, {"name": "Jane Doe"}],
            "venue": "AI Conference",
            "year": 2022,
            "publicationTypes": ["Conference"],
            "url": "https://semanticscholar.org/paper/123",
        }
        rec = s2_data_to_record(data)
        assert rec is not None
        assert rec.doi == "10.1234/s2test"
        assert rec.title == "Semantic Scholar Paper"
        assert rec.year == 2022
        assert len(rec.authors) == 2
        assert rec.authors[0]["family"] == "Smith"

    def test_external_ids_doi(self):
        data = {
            "externalIds": {"DOI": "10.1234/external"},
            "title": "Paper with External DOI",
            "authors": [],
        }
        rec = s2_data_to_record(data)
        assert rec is not None
        assert rec.doi == "10.1234/external"

    def test_single_name_author(self):
        data = {
            "doi": "10.1234/test",
            "title": "Test",
            "authors": [{"name": "Madonna"}],
        }
        rec = s2_data_to_record(data)
        assert rec is not None
        assert rec.authors[0]["family"] == "Madonna"
        assert rec.authors[0]["given"] == ""

    def test_publication_type_extracted(self):
        """Test that S2 types are normalized to standard types."""
        data = {
            "doi": "10.1234/test",
            "title": "Test",
            "publicationTypes": ["JournalArticle", "Review"],
        }
        rec = s2_data_to_record(data)
        assert rec is not None
        assert rec.type == "journal-article"  # Normalized from "journalarticle"

    def test_conference_type_normalized(self):
        """Test that S2 Conference type maps to proceedings-article."""
        data = {
            "doi": "10.1234/test",
            "title": "Test",
            "venue": "NeurIPS",
            "publicationTypes": ["Conference"],
        }
        rec = s2_data_to_record(data)
        assert rec is not None
        assert rec.type == "proceedings-article"

    def test_no_publication_types(self):
        data = {
            "doi": "10.1234/test",
            "title": "Test",
        }
        rec = s2_data_to_record(data)
        assert rec is not None
        assert rec.type == ""  # Empty string when no types provided

    def test_publication_venue_name(self):
        data = {
            "doi": "10.1234/test",
            "title": "Test",
            "publicationVenue": {"name": "Nature Machine Intelligence"},
            "venue": "Nat. Mach. Intell.",
        }
        rec = s2_data_to_record(data)
        assert rec is not None
        assert rec.journal == "Nature Machine Intelligence"


class TestDiskCache:
    """Tests for DiskCache class."""

    def test_cache_disabled_when_no_path(self):
        cache = DiskCache(None)
        cache.set("key", "value")
        assert cache.get("key") is None

    def test_cache_get_returns_none_for_missing_key(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        cache = DiskCache(str(cache_file))
        assert cache.get("nonexistent") is None

    def test_cache_set_and_get(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        cache = DiskCache(str(cache_file))
        cache.set("test_key", {"data": "value"})
        assert cache.get("test_key") == {"data": "value"}

    def test_cache_persists_to_disk(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        cache1 = DiskCache(str(cache_file))
        cache1.set("key", "persisted_value")

        # Create new cache instance reading from same file
        cache2 = DiskCache(str(cache_file))
        assert cache2.get("key") == "persisted_value"


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_allows_requests_below_limit(self):
        limiter = RateLimiter(100)
        # Should not block for 10 requests when limit is 100/min
        for _ in range(10):
            limiter.wait()
        # If we get here without blocking for long, test passes

    def test_rate_limiter_minimum_one(self):
        limiter = RateLimiter(0)
        assert limiter.req_per_min == 1
