"""Tests for bib_utils shared module.

These tests focus on API record conversion functions that are new to bib_utils.
Core utility functions are tested in test_utils.py.
"""

from __future__ import annotations

from bibtex_updater.utils import (
    DiskCache,
    RateLimiter,
    crossref_message_to_record,
    dblp_hit_to_candidate_record,
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

    def test_subtitle_joined_into_title(self):
        """ACM/IEEE split colon-titles across title/subtitle; the converter must
        re-join them (the CACM NeRF record: title=["NeRF"], subtitle=[...])."""
        msg = {
            "DOI": "10.1145/3503250",
            "type": "journal-article",
            "title": ["NeRF"],
            "subtitle": ["Representing scenes as neural radiance fields for view synthesis"],
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.title == "NeRF: Representing scenes as neural radiance fields for view synthesis"

    def test_subtitle_html_stripped_before_join(self):
        msg = {
            "DOI": "10.1234/test",
            "title": ["<i>Head</i>"],
            "subtitle": ["the <b>rest</b> of it"],
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.title == "Head: the rest of it"

    def test_no_subtitle_leaves_title_alone(self):
        msg = {
            "DOI": "10.1234/test",
            "title": ["Just a Title"],
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.title == "Just a Title"

    def test_empty_subtitle_list_leaves_title_alone(self):
        msg = {
            "DOI": "10.1234/test",
            "title": ["Just a Title"],
            "subtitle": [],
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.title == "Just a Title"

    def test_blank_subtitle_string_leaves_title_alone(self):
        msg = {
            "DOI": "10.1234/test",
            "title": ["Just a Title"],
            "subtitle": [""],
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.title == "Just a Title"

    def test_subtitle_already_in_title_not_duplicated(self):
        """Some publishers repeat the subtitle inside the full title; don't
        append it twice (case-insensitive containment check)."""
        msg = {
            "DOI": "10.1234/test",
            "title": ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"],
            "subtitle": ["Representing scenes as neural radiance fields for view synthesis"],
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.title == "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"

    def test_subtitle_without_title_ignored(self):
        msg = {
            "DOI": "10.1234/test",
            "subtitle": ["Orphan subtitle"],
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.title is None

    def test_created_date_is_last_resort(self):
        """``created`` is the DOI *deposit* date (years off for backfilled
        archives) -- ``issued`` must win over it."""
        msg = {
            "DOI": "10.1234/test",
            "issued": {"date-parts": [[1998]]},
            "created": {"date-parts": [[2015, 4, 1]]},
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.year == 1998

    def test_created_date_used_when_nothing_else(self):
        msg = {
            "DOI": "10.1234/test",
            "created": {"date-parts": [[2015, 4, 1]]},
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.year == 2015

    def test_published_print_still_wins_over_created(self):
        msg = {
            "DOI": "10.1234/test",
            "published-print": {"date-parts": [[2003]]},
            "created": {"date-parts": [[2011]]},
            "issued": {"date-parts": [[2004]]},
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.year == 2003


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

    def test_dblp_disambiguation_suffix_stripped_from_surname(self):
        """Regression: DBLP appends a 4-digit homonym suffix ("Yu Sun 0020",
        "Chuan Guo 0001"). It must be dropped so the family name is the real
        surname, not the number -- otherwise the author comparison scores a
        false author_mismatch against the correct bib entry.
        """
        hit = {
            "info": {
                "title": "On Calibration of Modern Neural Networks",
                "authors": {"author": ["Chuan Guo 0001", "Geoff Pleiss", "Yu Sun 0020", "Kilian Q. Weinberger"]},
                "venue": "Journal of Testing",
                "year": "2017",
                "doi": "10.1234/dblp.calib",
                "type": "Conference and Workshop Papers",
            }
        }
        rec = dblp_hit_to_record(hit)
        assert rec is not None
        families = [a["family"] for a in rec.authors]
        assert families == ["Guo", "Pleiss", "Sun", "Weinberger"]
        # The disambiguation digits move into the given name, not the surname.
        assert rec.authors[0] == {"given": "Chuan", "family": "Guo"}
        assert rec.authors[2] == {"given": "Yu", "family": "Sun"}

    def test_corr_venue_rejected_as_preprint(self):
        """Regression (F4): DBLP labels arXiv papers with venue "CoRR". Such a
        hit must be rejected (return None) so resolution falls through to the
        real published venue instead of accepting the preprint as published.
        """
        hit = {
            "info": {
                "title": "Some Preprint Indexed Under CoRR",
                "authors": {"author": ["Jane Doe"]},
                "venue": "CoRR",
                "year": "2024",
                "doi": "10.48550/arXiv.2401.00001",
                "type": "Journal Articles",
            }
        }
        assert dblp_hit_to_record(hit) is None


class TestDblpHitToCandidateRecord:
    """FIX E-DBLP: the permissive cascade-candidate converter must keep clean
    title+author hits that the strict resolver converter drops -- i.e. DOI-less
    conference papers and arXiv/CoRR copies -- so DBLP can contribute scorable
    candidates for ICML/ICLR/NeurIPS references.
    """

    def test_keeps_doi_less_conference_hit(self):
        """The exact failure case: a DOI-less, ee-less ICLR paper. The strict
        converter returns None here; the permissive one must keep it."""
        hit = {
            "info": {
                "title": "Context-Aware Sparse Deep Coordination Graphs",
                "authors": {"author": ["Tonghan Wang", "Liang Zeng"]},
                "venue": "ICLR",
                "year": "2022",
                "type": "Conference and Workshop Papers",
            }
        }
        assert dblp_hit_to_record(hit) is None  # strict drops it
        rec = dblp_hit_to_candidate_record(hit)  # permissive keeps it
        assert rec is not None
        assert rec.title == "Context-Aware Sparse Deep Coordination Graphs"
        assert rec.type == "proceedings-article"
        assert [a["family"] for a in rec.authors] == ["Wang", "Zeng"]

    def test_keeps_corr_arxiv_copy(self):
        """CoRR/arXiv hits are rejected by the strict converter but retained as
        candidates here (we don't discard the preprint copy outright)."""
        hit = {
            "info": {
                "title": "Some Paper Indexed Under CoRR",
                "authors": {"author": ["Jane Doe"]},
                "venue": "CoRR",
                "year": "2024",
                "doi": "10.48550/arXiv.2401.00001",
                "type": "Journal Articles",
            }
        }
        assert dblp_hit_to_record(hit) is None
        rec = dblp_hit_to_candidate_record(hit)
        assert rec is not None
        assert rec.title == "Some Paper Indexed Under CoRR"

    def test_strips_homonym_suffix(self):
        hit = {
            "info": {
                "title": "On Calibration of Modern Neural Networks",
                "authors": {"author": ["Chuan Guo 0001", "Yu Sun 0020"]},
                "venue": "ICML",
                "year": "2017",
                "type": "Conference and Workshop Papers",
            }
        }
        rec = dblp_hit_to_candidate_record(hit)
        assert rec is not None
        assert [a["family"] for a in rec.authors] == ["Guo", "Sun"]

    def test_rejects_missing_title(self):
        hit = {"info": {"authors": {"author": ["Jane Doe"]}, "venue": "ICML", "year": "2020"}}
        assert dblp_hit_to_candidate_record(hit) is None

    def test_rejects_no_authors(self):
        hit = {"info": {"title": "Authorless", "venue": "ICML", "year": "2020"}}
        assert dblp_hit_to_candidate_record(hit) is None

    def test_empty_hit(self):
        assert dblp_hit_to_candidate_record({}) is None


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
