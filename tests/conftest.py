"""Shared fixtures for bibtex_updater tests."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from bibtex_updater.updater import (
    Detector,
    FieldChecker,
    MissingFieldProcessor,
    PreprintDetection,
    Resolver,
    Updater,
)
from bibtex_updater.utils import HttpClient, PublishedRecord


@pytest.fixture
def make_entry():
    """Factory fixture for creating BibTeX entries."""

    def _make_entry(**kwargs) -> dict[str, Any]:
        entry = {
            "ENTRYTYPE": "article",
            "ID": kwargs.pop("ID", "testkey"),
            "title": "Example Title",
            "author": "Doe, Jane and Smith, John",
            "year": "2020",
        }
        entry.update(kwargs)
        return entry

    return _make_entry


@pytest.fixture
def arxiv_entry(make_entry):
    """A typical arXiv preprint entry."""
    return make_entry(
        ID="arxiv2020",
        title="Deep Learning for Everything",
        author="Smith, John and Doe, Jane",
        journal="arXiv preprint arXiv:2001.01234",
        url="https://arxiv.org/abs/2001.01234",
        year="2020",
    )


@pytest.fixture
def biorxiv_entry(make_entry):
    """A typical bioRxiv preprint entry."""
    return make_entry(
        ID="biorxiv2020",
        title="Novel Gene Discovery",
        author="Jones, Alice and Brown, Bob",
        journal="bioRxiv",
        doi="10.1101/2020.01.01.123456",
        year="2020",
    )


@pytest.fixture
def published_entry(make_entry):
    """A typical published journal article entry."""
    return make_entry(
        ID="published2021",
        title="Deep Learning for Everything",
        author="Smith, John and Doe, Jane",
        journal="Journal of Machine Learning",
        volume="42",
        pages="1-20",
        doi="10.1000/jml.2021.001",
        year="2021",
    )


@pytest.fixture
def detector():
    """Create a Detector instance."""
    return Detector()


@pytest.fixture
def updater():
    """Create an Updater instance with default settings."""
    return Updater(keep_preprint_note=False, rekey=False)


@pytest.fixture
def updater_with_note():
    """Create an Updater instance that keeps preprint notes."""
    return Updater(keep_preprint_note=True, rekey=False)


@pytest.fixture
def updater_with_rekey():
    """Create an Updater instance that regenerates keys."""
    return Updater(keep_preprint_note=False, rekey=True)


@pytest.fixture
def sample_published_record():
    """Create a sample PublishedRecord for testing."""
    return PublishedRecord(
        doi="10.1000/j.journal.2021.001",
        url="https://doi.org/10.1000/j.journal.2021.001",
        title="Deep Learning for Everything",
        authors=[
            {"given": "John", "family": "Smith"},
            {"given": "Jane", "family": "Doe"},
        ],
        journal="Journal of Machine Learning",
        publisher="Example Publisher",
        year=2021,
        volume="42",
        number="1",
        pages="1-20",
        type="journal-article",
        method="test",
        confidence=1.0,
    )


@pytest.fixture
def arxiv_detection():
    """Create a PreprintDetection for an arXiv entry."""
    return PreprintDetection(
        is_preprint=True,
        reason="url arXiv",
        arxiv_id="2001.01234",
        doi=None,
    )


@pytest.fixture
def biorxiv_detection():
    """Create a PreprintDetection for a bioRxiv entry."""
    return PreprintDetection(
        is_preprint=True,
        reason="preprint DOI pattern",
        arxiv_id=None,
        doi="10.1101/2020.01.01.123456",
    )


@pytest.fixture
def non_preprint_detection():
    """Create a PreprintDetection for a non-preprint entry."""
    return PreprintDetection(is_preprint=False)


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test")


class FakeHttpClient(HttpClient):
    """Fake HTTP client for testing without network calls."""

    def __init__(self):
        # Don't call parent __init__ to avoid setting up real HTTP
        pass

    def _request(self, method, url, params=None, accept=None):
        raise NotImplementedError("FakeHttpClient does not make real requests")


@pytest.fixture
def fake_http():
    """Create a fake HTTP client."""
    return FakeHttpClient()


class FakeResolver(Resolver):
    """Fake Resolver that returns predetermined results."""

    def __init__(self, result: PublishedRecord = None):
        self.logger = logging.getLogger("test")
        self.http = FakeHttpClient()
        self._result = result

    def resolve(self, entry, detection):
        return self._result


@pytest.fixture
def fake_resolver():
    """Factory fixture for creating fake resolvers."""

    def _create(result: PublishedRecord = None):
        return FakeResolver(result)

    return _create


# ------------- Field Checking Fixtures -------------


@pytest.fixture
def field_checker():
    """Create a FieldChecker instance."""
    return FieldChecker()


@pytest.fixture
def make_incomplete_entry(make_entry):
    """Factory fixture for creating incomplete BibTeX entries."""

    def _make_incomplete(**kwargs) -> dict[str, Any]:
        # Start with minimal entry, then apply overrides
        entry = {
            "ENTRYTYPE": kwargs.pop("ENTRYTYPE", "article"),
            "ID": kwargs.pop("ID", "incomplete"),
        }
        entry.update(kwargs)
        return entry

    return _make_incomplete


@pytest.fixture
def complete_article_entry(make_entry):
    """A complete article entry with all required and recommended fields."""
    return make_entry(
        ID="complete2020",
        title="Complete Article Title",
        author="Smith, John and Doe, Jane",
        journal="Journal of Complete Testing",
        year="2020",
        volume="42",
        number="1",
        pages="1-20",
        doi="10.1000/jct.2020.001",
        url="https://doi.org/10.1000/jct.2020.001",
    )


@pytest.fixture
def incomplete_article_entry(make_incomplete_entry):
    """An article entry missing required and recommended fields."""
    return make_incomplete_entry(
        ID="incomplete2020",
        title="Incomplete Article",
        # Missing: author, journal, year, volume, pages, doi, url
    )


@pytest.fixture
def field_processor_check_only(field_checker):
    """Create a MissingFieldProcessor in check-only mode."""
    return MissingFieldProcessor(
        checker=field_checker,
        filler=None,
        fill_mode="recommended",
        fill_enabled=False,
    )


# ------------- Fact Checker Fixtures -------------


@pytest.fixture
def make_verified_entry(make_entry):
    """Factory for creating entries that should verify successfully.

    These entries have real paper metadata that exists in external databases.
    """

    def _make(**kwargs):
        return make_entry(
            ID="vaswani2017attention",
            title="Attention Is All You Need",
            author="Vaswani, Ashish and Shazeer, Noam",
            journal="NeurIPS",
            year="2017",
            **kwargs,
        )

    return _make


@pytest.fixture
def make_hallucinated_entry(make_entry):
    """Factory for creating entries that appear to be fabricated.

    These entries have fake metadata that won't match any real papers.
    """

    def _make(**kwargs):
        return make_entry(
            ID="fakename2099",
            title="Quantum Widgets for AI: A Novel Approach",
            author="Fakename, Xavier and Imaginary, Yolanda",
            journal="Imaginary Journal of Nonexistent Science",
            year="2099",
            **kwargs,
        )

    return _make


@pytest.fixture
def make_mismatched_entry(make_entry):
    """Factory for creating entries with mismatched metadata.

    These entries have real paper metadata but with intentional mismatches.
    """

    def _make(mismatch_type="title", **kwargs):
        base = {
            "ID": "mismatched2020",
            "title": "Attention Is All You Need",
            "author": "Vaswani, Ashish",
            "journal": "NeurIPS",
            "year": "2017",
        }

        if mismatch_type == "title":
            base["title"] = "This Title Is Completely Wrong"
        elif mismatch_type == "author":
            base["author"] = "Wrong, Author and Different, Person"
        elif mismatch_type == "year":
            base["year"] = "2010"
        elif mismatch_type == "venue":
            base["journal"] = "Wrong Conference"

        base.update(kwargs)
        return make_entry(**base)

    return _make
