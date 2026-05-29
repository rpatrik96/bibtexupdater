"""Tests for the structured-author-name FPR fix in ``bibtex-check``.

Background. Author-surname comparison used to re-derive a surname from a
*flattened* "Given Family" string by taking the LAST token
(``last_name_from_person``). For family-first names (Chinese/Korean) sources
disagree on order: an entry "Chen Xing" (family-first) reduced to ``xing`` while
the API record for the same person reduced to ``chen`` -> a FALSE
AUTHOR_MISMATCH. Crossref returns AUTHORITATIVE separate ``given``/``family``
fields, so we now:

* ``PublishedRecord.surname_keys`` trusts an authoritative ``family`` verbatim
  (normalized, no last-token reduction) when ``structured_names`` is True, and
  falls back to ``last_name_from_person`` on the flat name otherwise.
* ``entry_surnames_against_structured`` disambiguates an order-ambiguous
  comma-less entry name using the structured record's family set.
* The cascade re-fetches the cited paper from a STRUCTURED source (Crossref by
  DOI, else a Crossref title search) before emitting an AUTHOR_MISMATCH that was
  driven by an UNSTRUCTURED candidate (S2 flat names / DBLP / OpenAlex).

Hard constraint: these changes only fix WHICH token is the surname. A genuinely
different author set, a swapped lead author, or placeholder/fabricated authors
must STILL produce AUTHOR_MISMATCH -- order-sensitivity and the match threshold
are untouched. All tests are hermetic (mocked API returns, no network).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from bibtex_updater.fact_checker import (
    CrossrefClient,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    FactCheckStatus,
    SemanticScholarClient,
)
from bibtex_updater.matching import MatchOutcome, symmetric_author_match
from bibtex_updater.utils import (
    PublishedRecord,
    crossref_message_to_record,
    entry_surnames_against_structured,
    s2_data_to_record,
)


@pytest.fixture
def logger():
    return logging.getLogger("test_structured_author_names")


@pytest.fixture
def empty_http():
    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return mock


@pytest.fixture
def dead_sources(empty_http):
    return (
        CrossrefClient(empty_http),
        DBLPClient(empty_http),
        SemanticScholarClient(empty_http),
    )


def _crossref_message(title: str, doi: str, authors: list[dict]) -> dict:
    msg = {
        "DOI": doi,
        "type": "proceedings-article",
        "title": [title],
        "author": authors,
        "container-title": ["Some Conference"],
        "issued": {"date-parts": [[2024]]},
    }
    return msg


# ------------- (b) surname_keys prefers the structured family -------------


class TestSurnameKeysUsesFamily:
    def test_structured_family_not_last_token(self):
        """A structured record trusts ``family`` -- "Chen" survives, not "xing"."""
        rec = PublishedRecord(
            doi="10.1/x",
            authors=[{"given": "Xing", "family": "Chen"}],
            structured_names=True,
        )
        assert rec.surname_keys() == ["chen"]

    def test_unstructured_falls_back_to_last_token(self):
        """No authoritative family split -> reconstruct + last_name_from_person."""
        # Simulate a flat name an unstructured source synthesized: the family
        # slot holds the last token of the flat string.
        rec = PublishedRecord(
            doi="10.1/x",
            authors=[{"given": "Xing", "family": "Chen"}],
            structured_names=False,
        )
        # Flat "Xing Chen" -> last token -> "chen" (the heuristic guess).
        assert rec.surname_keys() == ["chen"]

    def test_crossref_converter_sets_structured(self):
        rec = crossref_message_to_record(
            {"DOI": "10.1/x", "title": ["t"], "author": [{"given": "Xing", "family": "Chen"}]}
        )
        assert rec is not None
        assert rec.structured_names is True
        assert rec.surname_keys() == ["chen"]

    def test_crossref_literal_only_is_not_structured(self):
        """A literal-only Crossref author was split by us, so it is unstructured."""
        rec = crossref_message_to_record(
            {"DOI": "10.1/x", "title": ["t"], "author": [{"literal": "Xing Chen"}]}
        )
        assert rec is not None
        assert rec.structured_names is False

    def test_s2_flat_name_is_not_structured(self):
        rec = s2_data_to_record(
            {"doi": "10.1/x", "title": "t", "authors": [{"name": "Xing Chen"}], "venue": "J"}
        )
        assert rec is not None
        assert rec.structured_names is False

    def test_structured_multitoken_western_family_symmetric(self):
        """A structured multi-token family ("van den Oord") still reduces to the
        distinctive last token "oord", matching the entry side."""
        rec = PublishedRecord(
            doi="10.1/y",
            authors=[{"given": "Aaron", "family": "van den Oord"}],
            structured_names=True,
        )
        assert rec.surname_keys() == ["oord"]


# ------------- (a) CJK flip: entry "Chen Xing" vs structured record -------------


class TestCjkFlipMatches:
    def test_entry_family_first_matches_structured_record(self):
        """Entry "Chen Xing" (family-first, comma-less) vs structured record
        ``{given: Xing, family: Chen}`` -> MATCH (was a false MISMATCH)."""
        rec = PublishedRecord(
            doi="10.1/x",
            authors=[{"given": "Xing", "family": "Chen"}],
            structured_names=True,
        )
        family_keys = set(rec.surname_keys(limit=10_000))
        entry_keys = entry_surnames_against_structured("Chen Xing", family_keys, limit=10_000)
        api_keys = rec.surname_keys(limit=10_000)
        assert entry_keys == ["chen"]
        result = symmetric_author_match(entry_keys, api_keys, threshold=0.8)
        assert result.outcome is MatchOutcome.MATCH

    def test_naive_last_token_would_have_mismatched(self):
        """Documents the bug: without disambiguation "Chen Xing" -> "xing"."""
        from bibtex_updater.utils import authors_last_names

        rec = PublishedRecord(
            doi="10.1/x", authors=[{"given": "Xing", "family": "Chen"}], structured_names=True
        )
        naive_entry = authors_last_names("Chen Xing", limit=10_000)  # ["xing"]
        result = symmetric_author_match(naive_entry, rec.surname_keys(10_000), threshold=0.8)
        assert result.outcome is MatchOutcome.MISMATCH  # the old false positive

    def test_comma_form_entry_already_unambiguous(self):
        """A comma-form entry "Chen, Xing" is already family-first -> "chen"."""
        rec = PublishedRecord(
            doi="10.1/x", authors=[{"given": "Xing", "family": "Chen"}], structured_names=True
        )
        family_keys = set(rec.surname_keys(10_000))
        keys = entry_surnames_against_structured("Chen, Xing", family_keys, limit=10_000)
        assert keys == ["chen"]


# ------------- (c) cascade fallback: S2 flat mismatch + matching DOI -------------


class TestCascadeStructuredFallback:
    def _cjk_entry(self) -> dict[str, str]:
        # Family-first comma-less entry; DOI present.
        return {
            "ID": "cjk2024",
            "ENTRYTYPE": "inproceedings",
            "title": "A Family-First Authored Paper on Representation Learning",
            "author": "Chen Xing and Wang Lei",
            "doi": "10.1/cjkpaper",
            "year": "2024",
        }

    def test_s2_mismatch_suppressed_by_structured_doi(self, dead_sources, logger):
        """An S2 flat-name candidate that would AUTHOR_MISMATCH is vetted against
        the structured Crossref record for the SAME DOI; the structured authors
        match -> mismatch suppressed (entry verifies / not AUTHOR_MISMATCH)."""
        crossref, dblp, s2 = dead_sources
        entry = self._cjk_entry()

        # Structured Crossref record (authoritative given/family) for this DOI.
        structured_authors = [
            {"given": "Xing", "family": "Chen"},
            {"given": "Lei", "family": "Wang"},
        ]
        cr_msg = _crossref_message(entry["title"], entry["doi"], structured_authors)
        crossref.get_by_doi = MagicMock(return_value=cr_msg)
        # Crossref title search returns nothing (force the candidate to come from
        # S2 with a flat, order-ambiguous name list).
        crossref.search = MagicMock(return_value=[])
        # S2 returns the same paper with FLAT names in family-first order, which
        # the last-token heuristic mis-parses, producing the false mismatch.
        s2.search = MagicMock(
            return_value=[
                {
                    "title": entry["title"],
                    "doi": entry["doi"],
                    "venue": "Some Conference",
                    "year": 2024,
                    "authors": [{"name": "Xing Chen"}, {"name": "Lei Wang"}],
                }
            ]
        )

        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)
        result = checker.check_entry(entry)

        assert result.status != FactCheckStatus.AUTHOR_MISMATCH
        # The structured re-fetch via the entry's DOI was consulted.
        crossref.get_by_doi.assert_called_with(entry["doi"])

    def test_cascade_fallback_isolated_from_presearch_doi_check(self, dead_sources, logger):
        """With the pre-search DOI-consistency guard DISABLED, only the cascade
        ``_structured_author_recheck`` can suppress the S2-driven mismatch -- so
        this proves the cascade fallback works on its own."""
        crossref, dblp, s2 = dead_sources
        entry = self._cjk_entry()
        structured_authors = [
            {"given": "Xing", "family": "Chen"},
            {"given": "Lei", "family": "Wang"},
        ]
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(entry["title"], entry["doi"], structured_authors)
        )
        crossref.search = MagicMock(return_value=[])
        s2.search = MagicMock(
            return_value=[
                {
                    "title": entry["title"],
                    "doi": entry["doi"],
                    "venue": "Some Conference",
                    "year": 2024,
                    "authors": [{"name": "Xing Chen"}, {"name": "Lei Wang"}],
                }
            ]
        )
        config = FactCheckerConfig(check_doi_consistency=False)
        checker = FactChecker(crossref, dblp, s2, config, logger)
        result = checker.check_entry(entry)

        assert result.status != FactCheckStatus.AUTHOR_MISMATCH
        crossref.get_by_doi.assert_called_with(entry["doi"])

    def test_fallback_keeps_mismatch_when_structured_also_differs(self, dead_sources, logger):
        """If the structured record genuinely disagrees, AUTHOR_MISMATCH stands."""
        crossref, dblp, s2 = dead_sources
        entry = self._cjk_entry()

        # Structured record lists a DIFFERENT lead author than the entry.
        different_authors = [
            {"given": "Alice", "family": "Smith"},
            {"given": "Bob", "family": "Jones"},
        ]
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(entry["title"], entry["doi"], different_authors)
        )
        crossref.search = MagicMock(return_value=[])
        s2.search = MagicMock(
            return_value=[
                {
                    "title": entry["title"],
                    "doi": entry["doi"],
                    "venue": "Some Conference",
                    "year": 2024,
                    "authors": [{"name": "Xing Chen"}, {"name": "Lei Wang"}],
                }
            ]
        )

        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)
        result = checker.check_entry(entry)

        assert result.status == FactCheckStatus.AUTHOR_MISMATCH

    def test_fallback_uses_title_search_when_no_doi(self, dead_sources, logger):
        """No DOI -> the re-check uses a confident-title Crossref search."""
        crossref, dblp, s2 = dead_sources
        entry = self._cjk_entry()
        del entry["doi"]

        structured_authors = [
            {"given": "Xing", "family": "Chen"},
            {"given": "Lei", "family": "Wang"},
        ]
        cr_msg = _crossref_message(entry["title"], "10.1/found", structured_authors)
        crossref.get_by_doi = MagicMock(return_value=None)
        # Title search returns the structured record only on the second call
        # (the cascade's own first call vs. the re-check's). Simpler: always
        # return the structured record from search; the cascade scores it but the
        # S2 flat-name candidate is what wins author scoring in this construction.
        crossref.search = MagicMock(return_value=[cr_msg])
        s2.search = MagicMock(
            return_value=[
                {
                    "title": entry["title"],
                    "venue": "Some Conference",
                    "year": 2024,
                    "authors": [{"name": "Xing Chen"}, {"name": "Lei Wang"}],
                }
            ]
        )

        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)
        result = checker.check_entry(entry)

        assert result.status != FactCheckStatus.AUTHOR_MISMATCH


# ------------- (d) GUARDS: hallucination detection preserved -------------


class TestGuardsPreserved:
    def test_genuinely_different_authors_still_mismatch(self, dead_sources, logger):
        """A different author set is NOT rescued by the structured re-fetch."""
        crossref, dblp, s2 = dead_sources
        entry = {
            "ID": "diff2024",
            "ENTRYTYPE": "inproceedings",
            "title": "A Paper With Completely Fabricated Authors",
            "author": "Fakename, Imaginary and Madeup, Person",
            "doi": "10.1/realpaper",
            "year": "2024",
        }
        real_authors = [
            {"given": "Real", "family": "Researcher"},
            {"given": "Actual", "family": "Scientist"},
        ]
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(entry["title"], entry["doi"], real_authors)
        )
        crossref.search = MagicMock(
            return_value=[_crossref_message(entry["title"], entry["doi"], real_authors)]
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)
        assert result.status == FactCheckStatus.AUTHOR_MISMATCH

    def test_swapped_first_author_still_mismatch(self):
        """A swapped (different) lead author is a real mismatch -- the structured
        family set cannot order-insensitively rescue a wrong lead author."""
        rec = PublishedRecord(
            doi="10.1/x",
            authors=[
                {"given": "Xing", "family": "Chen"},
                {"given": "Lei", "family": "Wang"},
            ],
            structured_names=True,
        )
        family_keys = set(rec.surname_keys(10_000))
        # Entry lists a DIFFERENT person ("Wei Liu") as the lead author.
        entry_keys = entry_surnames_against_structured("Wei Liu and Xing Chen", family_keys, 10_000)
        result = symmetric_author_match(entry_keys, rec.surname_keys(10_000), threshold=0.8)
        assert result.outcome is MatchOutcome.MISMATCH

    def test_placeholder_authors_still_mismatch(self, dead_sources, logger):
        """Placeholder/fabricated authors on a real DOI still flag."""
        crossref, dblp, s2 = dead_sources
        entry = {
            "ID": "placeholder2024",
            "ENTRYTYPE": "inproceedings",
            "title": "Real Paper Title Carrying Placeholder Authors",
            "author": "Author, First and Author, Second",
            "doi": "10.1/realdoi",
            "year": "2024",
        }
        real_authors = [
            {"given": "Genuine", "family": "Person"},
            {"given": "Another", "family": "Individual"},
        ]
        crossref.get_by_doi = MagicMock(
            return_value=_crossref_message(entry["title"], entry["doi"], real_authors)
        )
        crossref.search = MagicMock(
            return_value=[_crossref_message(entry["title"], entry["doi"], real_authors)]
        )
        checker = FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)

        result = checker.check_entry(entry)
        assert result.status == FactCheckStatus.AUTHOR_MISMATCH

    def test_different_author_not_rescued_by_family_set(self):
        """``entry_surnames_against_structured`` never rescues a foreign name:
        "John Smith" against a {"chen"} family set stays "smith"."""
        keys = entry_surnames_against_structured("John Smith", {"chen"}, limit=10_000)
        assert keys == ["smith"]
